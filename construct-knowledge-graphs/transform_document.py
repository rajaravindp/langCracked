from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

def transform_document(document: str) -> LLMGraphTransformer:
    """
    Transform a document into graph nodes and edges using LLMGraphTransformer.

    Args:
        document (str): The input document to be transformed.

    Returns:
        LLMGraphTransformer: An instance of LLMGraphTransformer with the transformed document.
    """

    # Initialize the LLM for graph transformation
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 


    # Define the allowed nodes, properties, and relationships for the graph
    # llm_transformer = LLMGraphTransformer(llm=llm)
    llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        # People and Beings
        "Person",           # Frodo, Bilbo, Gandalf, Gollum, etc.
        "Race",            # Hobbits, Elves, Dwarves, Men, Orcs
        "Family",          # Baggins, Took, Brandybuck families
        
        # Places and Geography
        "Region",          # The Shire, Eriador, Middle-earth, Gondor
        "Settlement",      # Hobbiton, Bree, Michel Delving, Bag End
        "GeographicFeature", # Misty Mountains, Brandywine River, Far Downs
        
        # Objects and Artifacts
        "MagicalItem",     # The One Ring, Sting sword, mithril coat
        "Book",           # Red Book of Westmarch, The Hobbit
        "Plant",          # Pipe-weed, Old Toby, Longbottom Leaf
        
        # Organizations and Roles
        "Office",         # Thain, Mayor, Shirriff
        "Institution",    # Shire-government, Michel Delving Mathom-house
        
        # Time and Events
        "Event",          # Battle of Greenfields, War of the Ring
        "TimePeriod"      # Third Age, Shire-reckoning years
    ],
    node_properties=[
        # Person properties
        "age",                    # Bilbo was 111, Frodo was 50
        "birth_year",            # S.R. years or Third Age years
        "race",                  # Hobbit, Wizard, etc.
        "height",                # "between two and four feet"
        "family_name",           # Baggins, Took, Brandybuck
        "title",                 # "The Took", "Thain", "Ring-bearer"
        
        # Place properties
        "size",                  # "Forty leagues", "fifty from north to south"
        "population",            # Various settlements
        "founding_year",         # S.R. 1 for The Shire
        "location_description",  # "North-West of the Old World"
        
        # Object properties
        "material",              # "gold", "mithril", "clay or wood"
        "origin",                # Where items came from
        "power",                 # "makes wearer invisible"
        "creator",               # Who made the item
        
        # Event properties
        "year",                  # S.R. 1147 for Battle of Greenfields
        "participants",          # Who was involved
        "outcome",               # Result of the event
        
        # General properties
        "description",           # Physical or functional description
        "importance",            # Significance in the story
        "status"                 # Current state or condition
    ],
    allowed_relationships=[
        # Geographic relationships
        "LOCATED_IN",      # Hobbiton LOCATED_IN The Shire
        "BORDERS",         # The Shire BORDERS Bree-land
        "FLOWS_THROUGH",   # Brandywine River FLOWS_THROUGH The Shire
        
        # Family and social relationships
        "MEMBER_OF",       # Frodo MEMBER_OF Baggins family
        "RELATED_TO",      # Bilbo RELATED_TO Frodo
        "DESCENDANT_OF",   # Bandobras Took DESCENDANT_OF Isumbras
        
        # Ownership and possession
        "OWNS",           # Bilbo OWNS The One Ring
        "INHERITS",       # Frodo INHERITS Bag End
        "LIVES_IN",       # Frodo LIVES_IN Bag End
        
        # Political and social roles
        "HOLDS_OFFICE",   # Took family HOLDS_OFFICE Thain
        "RULES",          # Thain RULES The Shire
        "SERVES",         # Shirriffs SERVES The Shire
        
        # Interactions and events
        "MEETS",          # Bilbo MEETS Gollum
        "TRAVELS_TO",     # Bilbo TRAVELS_TO Erebor
        "PARTICIPATES_IN", # Bandobras PARTICIPATES_IN Battle of Greenfields
        "CREATES",        # Tobold Hornblower CREATES pipe-weed cultivation
        
        # Knowledge and documentation
        "WRITES",         # Bilbo WRITES Red Book
        "RECORDS",        # Red Book RECORDS War of the Ring
        "MENTIONS",       # The Hobbit MENTIONS Bilbo's journey
        
        # Racial and cultural relationships
        "BELONGS_TO_RACE", # Frodo BELONGS_TO_RACE Hobbit
        "ASSOCIATED_WITH", # Pipe-weed ASSOCIATED_WITH Hobbits
        "INFLUENCED_BY",   # Hobbit architecture INFLUENCED_BY Dwarves
        
        # Temporal relationships
        "OCCURS_DURING",  # Battle of Greenfields OCCURS_DURING S.R. 1147
        "PRECEDES",       # The Hobbit PRECEDES Lord of the Rings
        "FOUNDED_IN"      # The Shire FOUNDED_IN S.R. 1
    ]
)
    
    # Convert the document to graph documents
    # This will create nodes and relationships based on the document content
    return llm_transformer.convert_to_graph_documents([Document(page_content=document)])
