import os
from vespa.package import Schema, Document, Field, FieldSet, HNSW
from vespa.package import ApplicationPackage
from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking

# Define Vespa schema and ranking profiles
colpali_schema = Schema(
    name="pdf_page",
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
    name="default",
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
vespa_app_name = "codersociety"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name,
    schema=[colpali_schema]
)

# Save the application package to the 'vespa_app_name' directory
vespa_application_package.to_files(vespa_app_name)
