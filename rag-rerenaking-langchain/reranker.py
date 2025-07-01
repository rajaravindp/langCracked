import boto3

region = boto3.Session().region_name
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime',region_name=region)
# model_arn = f"arn:aws:bedrock:{region}:852234679667:foundation-model/cohere.rerank-v3-5:"

def reranker(text_query: str, document_sources: list, num_results: int,  model_id: str = "cohere.rerank-english-v3.0") -> list:
    response = bedrock_agent_runtime.rerank(
        queries=[
            {
                "type": "TEXT",
                "textQuery": {
                    "text": text_query
                }
            }
        ],
        sources=document_sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": num_results,
                "modelConfiguration": {
                    "modelArn": model_id,
                }
            }
        }
    )
    return response['results']