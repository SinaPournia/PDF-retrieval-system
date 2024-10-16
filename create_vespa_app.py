import os
import subprocess
import dotenv
from vespa.package import Schema, Document, Field, FieldSet, HNSW
from vespa.package import ApplicationPackage
from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking
from config import Settings  
from datetime import datetime, timedelta




settings = Settings()

def create_and_save_vespa_schema():
    # Define Vespa schema and ranking profiles
    colpali_schema = Schema(
        name=settings.vespa_app_name,  # Use vespa app name from settings
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"], match=["word"]),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(name="title", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(name="text", type="string", indexing=["index"], match=["text"], index="enable-bm25"),
                Field(
                    name="embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=["attribute", "index"],
                    ann=HNSW(distance_metric="hamming", max_links_per_node=32, neighbors_to_explore_at_insert=400),
                )
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["title", "text"])]
    )

    # Create ranking profiles
    colpali_profile = RankProfile(
        name=settings.ranking_profile_name,  # Use ranking profile name from settings
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(
                name="bm25_score", expression="bm25(title) + bm25(text)"
            )
        ],
        first_phase=FirstPhaseRanking(expression="bm25_score"),
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=100)
    )
    colpali_schema.add_rank_profile(colpali_profile)

    # Create Vespa application package
    vespa_app_name = settings.vespa_app_name  # Use Vespa app name from settings
    vespa_application_package = ApplicationPackage(
        name=vespa_app_name,
        schema=[colpali_schema]
    )

    # Save the application package to the 'vespa_app_name' directory
    vespa_application_package.to_files(vespa_app_name)

    return vespa_app_name


def create_validation_overrides(app_dir):
    # Get tomorrow's date in the desired format (YYYY-MM-DD)
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    validation_file = os.path.join(app_dir, "validation-overrides.xml")
    
    if not os.path.exists(validation_file):
        with open(validation_file, "w") as f:
            f.write(f"""
            <validation-overrides>
              <allow until='{tomorrow_date}'>schema-removal</allow>
              <allow until='{tomorrow_date}'>content-cluster-removal</allow>  
            </validation-overrides>
            """)

def deploy_vespa_application(app_dir):
    # Create validation overrides
    create_validation_overrides(app_dir)
    
    # Deploy the application using Vespa CLI
    try:
        deploy_command = ["vespa", "deploy", app_dir]  
        result = subprocess.run(deploy_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = result.stdout.decode('utf-8')
        stderr = result.stderr.decode('utf-8')
        
        if stdout:
            print(f"Vespa deploy output: {stdout}")
        if stderr:
            print(f"Vespa deploy error: {stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error during deployment: {e.stderr.decode('utf-8')}")



if __name__ == "__main__":
    # Step 1: Create and save Vespa schema
    app_directory = create_and_save_vespa_schema()

    # Step 2: Deploy the Vespa application to the local container
    deploy_vespa_application(app_directory)
