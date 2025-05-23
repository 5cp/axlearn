# Copyright © 2024 Apple Inc.

"""Utilities to provision TPU node pools."""

import hashlib
import io
import os
import time
from typing import Optional

from absl import flags, logging

from axlearn.cloud.common.bastion import _BASTION_SERIALIZED_JOBSPEC_ENV_VAR, deserialize_jobspec
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import GKEJob, TPUGKEJob
from axlearn.cloud.gcp.job_flink import FlinkTPUGKEJob
from axlearn.cloud.gcp.jobset_utils import TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import (
    construct_node_pool_name,
    create_node_pools,
    delete_node_pools,
)
from axlearn.cloud.gcp.system_characteristics import USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS
from axlearn.cloud.gcp.tpu import infer_tpu_type
from axlearn.common.config import REQUIRED, Configurable, Required, config_class

FLAGS = flags.FLAGS

# TODO(muyang_yu): avoid listing job types one by one.
_PRE_PROVISIONER_SUPPORTED_JOBS = (TPUGKEJob, FlinkTPUGKEJob)
_INFERENCE_JOBS = (FlinkTPUGKEJob,)


class NodePoolProvisioner(Configurable):
    """Node pool provisioner."""

    @config_class
    class Config(Configurable.Config):
        """Configures node pool provisioning.

        Attributes:
            project: GCP Project name.
            zone: GCP zone name.
            cluster: K8s cluster name.
            name: The name of the provisioner.
            service_account_email: Service account email for node pools.
            retry_interval: Number of seconds to retry node pool creation or deletion.
            wait_timeout: Number of seconds to wait for node pool creation or deletion.
        """

        project: Required[str] = REQUIRED
        zone: Required[str] = REQUIRED
        cluster: Required[str] = REQUIRED
        name: Required[str] = REQUIRED
        # If not none, node pools will be created with the service account email.
        service_account_email: Optional[str] = None
        retry_interval: int = 30
        wait_timeout: int = 30 * 60

    @classmethod
    def from_flags(cls, fv: flags.FlagValues) -> Config:
        cfg = super().default_config()

        cfg.project = gcp_settings("project", fv=fv)
        cfg.zone = gcp_settings("zone", fv=fv)
        cfg.cluster = gcp_settings("gke_cluster", fv=fv)
        cfg.service_account_email = gcp_settings("service_account_email", required=False, fv=fv)

        return cfg

    def create_for(self, job: GKEJob):
        """Creates node pools for the job."""
        raise NotImplementedError(type(self))

    def delete_for(self, job: GKEJob):
        """Deletes node pools for the job."""
        raise NotImplementedError(type(self))


class TPUNodePoolProvisioner(NodePoolProvisioner):
    """TPU node pool provisioner."""

    def create_for(self, job: TPUGKEJob):
        """Creates named node pools for the job."""

        # TODO(markblee,ethanli,muyang_yu): Refactor so we do not need to make assumptions about
        # TPUGKEJob implementation and internals.
        if not isinstance(job, _PRE_PROVISIONER_SUPPORTED_JOBS):
            raise TypeError(f"Expected {_PRE_PROVISIONER_SUPPORTED_JOBS}, got {type(job)}.")

        cfg: TPUNodePoolProvisioner.Config = self.config
        job_cfg: TPUGKEJob.Config = job.config
        builder_cfg: TPUReplicatedJob.Config = job_cfg.builder
        acc_cfg = builder_cfg.accelerator
        reservation = builder_cfg.reservation
        location_hint = builder_cfg.location_hint
        enable_tpu_ici_resiliency = builder_cfg.enable_tpu_ici_resiliency
        enable_tpu_smart_repair = builder_cfg.enable_tpu_smart_repair
        tpu_type = infer_tpu_type(acc_cfg.instance_type)
        job_sys_property = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[tpu_type]
        num_node_pools = acc_cfg.num_replicas

        tier = os.environ.get("BASTION_TIER", 0)
        if tier == "0" and reservation is not None:
            logging.info("Found tier=%s in env. Using reservation=%s", tier, reservation)
            use_spot_vm = False
        else:
            logging.info("Found tier=%s in env. Using spot quota", tier)
            use_spot_vm = True
            reservation = None

        job_priority = None
        if os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR):
            spec = deserialize_jobspec(
                io.StringIO(os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR))
            )
            job_priority = spec.metadata.priority

        node_pool_names = []
        additional_labels_list = []
        for i in range(num_node_pools):
            node_pool_names.append(
                construct_node_pool_name(
                    jobset_namespace=job_cfg.namespace, jobset_name=job_cfg.name, index=i
                )
            )
            # This is required because the jobset-controller-manager will
            # inject this node-selector to pods.
            # https://github.com/nstogner/jobset/commit/59c93ca5b0df408b7b6f19edcbd255079c8e0b2a
            # TODO(ethanli): remove this hack once jobset-controller-manager
            #  supports disabling node-selector injections
            job_key = hashlib.sha1(
                f"{job_cfg.namespace}/{job_cfg.name}-job-{i}".encode()
            ).hexdigest()
            additional_labels = {"job-key": job_key}

            # Populate job-priority label to nodes.
            if job_priority is not None:
                additional_labels.update({"job-priority": str(job_priority)})

            if enable_tpu_smart_repair:
                additional_labels.update({"cloud.google.com/gke-tpu-auto-restart": "true"})

            additional_labels_list.append(additional_labels)

        start_time = time.perf_counter()
        topology = None if isinstance(job, _INFERENCE_JOBS) else job_sys_property.topology
        create_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            pre_provisioner_id=cfg.name,
            num_nodes_per_pool=job_sys_property.vms_per_slice,
            machine_type=job_sys_property.gce_machine_type,
            topology=topology,
            use_spot_vm=use_spot_vm,
            reservation=reservation,
            location_hint=location_hint,
            enable_tpu_ici_resiliency=enable_tpu_ici_resiliency,
            service_account_email=cfg.service_account_email,
            additional_labels_list=additional_labels_list,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout,
        )

        elapsed_time = time.perf_counter() - start_time
        logging.info(
            "%s node pools for %s creation took %s seconds", num_node_pools, cfg.name, elapsed_time
        )

    def delete_for(self, job: TPUGKEJob):
        """Deletes node pools of the job."""

        if not isinstance(job, _PRE_PROVISIONER_SUPPORTED_JOBS):
            raise TypeError(f"Expected {_PRE_PROVISIONER_SUPPORTED_JOBS}, got {type(job)}.")

        cfg: TPUNodePoolProvisioner.Config = self.config
        job_cfg: TPUGKEJob.Config = job.config
        builder_cfg: TPUReplicatedJob.Config = job_cfg.builder
        num_node_pools = builder_cfg.accelerator.num_replicas

        node_pool_names = []

        for i in range(num_node_pools):
            node_pool_names.append(
                construct_node_pool_name(
                    jobset_namespace=job_cfg.namespace, jobset_name=job_cfg.name, index=i
                )
            )

        start_time = time.perf_counter()
        delete_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout,
        )

        elapsed_time = time.perf_counter() - start_time
        logging.info("Node pool deletion took %s seconds", elapsed_time)
