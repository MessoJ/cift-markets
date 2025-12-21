import argparse

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.entities import AmlCompute, BuildContext, Environment
from azure.identity import DefaultAzureCredential


def main():
    parser = argparse.ArgumentParser(description="Submit CIFT Walkforward Training to Azure ML")
    parser.add_argument("--subscription-id", required=True, help="Azure Subscription ID")
    parser.add_argument("--resource-group", required=True, help="Azure Resource Group Name")
    parser.add_argument("--workspace-name", required=True, help="Azure ML Workspace Name")
    parser.add_argument("--compute-name", default="cpu-cluster", help="Azure ML Compute Cluster Name")
    parser.add_argument("--data-path", required=True, help="Path to data file (local or cloud URI)")
    args = parser.parse_args()

    # Connect to Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # Create Compute Cluster if not exists
    try:
        ml_client.compute.get(args.compute_name)
        print(f"Compute target '{args.compute_name}' already exists.")
    except Exception:
        print(f"Creating compute target '{args.compute_name}'...")
        compute_target = AmlCompute(
            name=args.compute_name,
            type="amlcompute",
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(compute_target).result()

    # Define the job
    # We use the Dockerfile.training to build the environment
    job = command(
        code="./",  # Upload current directory
        command="""python -m cift.cli walkforward \
            --data-path ${{inputs.data}} \
            --model xgboost \
            --use-fracdiff --fracdiff-d 0.4 \
            --use-vol-features \
            --use-ta-features \
            --use-micro-features \
            --use-meta-labeling \
            --vol-target 0.15 \
            --tune-model \
            --n-splits 5 \
            --n-trials 20
        """,
        inputs={
            "data": Input(
                type="uri_file",
                path=args.data_path,
            )
        },
        environment=Environment(
            build=BuildContext(path="./", dockerfile_path="Dockerfile.training"),
            name="cift-training-env",
            version="1.0"
        ),
        compute=args.compute_name,
        display_name="cift-walkforward-brutal",
        experiment_name="cift-sharpe-optimization",
    )

    # Submit
    print("Submitting job...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted. Studio URL: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
