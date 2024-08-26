from rag_core.rag_ops import RAGOperations


rag_operations = RAGOperations()

while True:
    input_query = input("Query (type 'bye' or 'exit' to quit the program ):")
    if input_query.lower() == 'bye' or input_query.lower() == 'exit':
        break
    result = rag_operations.start_conversation(user_query=input_query)
    print(f"Result: {result}")