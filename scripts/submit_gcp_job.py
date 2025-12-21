import argparse

from google.cloud import aiplatform


def submit_job(
    project_id: str,
    location: str,
    display_name: str,
    container_image_uri: str,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
):
    """
    Submits a Custom Training Job to Google Cloud Vertex AI.
    """
    print(f"Initializing Vertex AI for project {project_id} in {location}...")
    aiplatform.init(project=project_id, location=location)

    print(f"Creating CustomContainerTrainingJob: {display_name}")
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_image_uri,
        # command=["python", "-m", "cift.ml.evaluation.walkforward"], # Optional override
    )

    print(f"Submitting job with {machine_type} and {accelerator_count}x {accelerator_type}...")
    # Note: For TPUs, use accelerator_type="TPU_V3" and appropriate machine type

    try:
        job.run(
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            replica_count=1,
            sync=False, # Return immediately
        )
        print("Job submitted successfully.")
        print(f"Resource Name: {job.resource_name}")
        print(f"Dashboard: https://console.cloud.google.com/vertex-ai/training/training-pipelines?project={project_id}")
    except Exception as e:
        print(f"Failed to submit job: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit training job to Google Cloud Vertex AI")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Region (e.g., us-central1)")
    parser.add_argument("--image", required=True, help="Docker image URI (e.g., gcr.io/my-project/cift-training:latest)")
    parser.add_argument("--name", default="cift-training-job", help="Job display name")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type (e.g., n1-standard-4, n1-highmem-8)")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4", help="Accelerator type (NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, TPU_V2, TPU_V3)")
    parser.add_argument("--accelerator-count", type=int, default=1, help="Number of accelerators")

    args = parser.parse_args()

    submit_job(
        project_id=args.project_id,
        location=args.location,
        display_name=args.name,
        container_image_uri=args.image,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count
    )
