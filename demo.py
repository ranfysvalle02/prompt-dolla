import re
import json
import ollama
desiredModel='llama3.2:3b'

# Expanded dataset: a list of movie comments with a mixed bag of entries
DATASET = [
    {
        "_id": 1,
        "user": "Alice",
        "movie": "Inception",
        "rating": 5,
        "comment": "This movie is absolutely amazing! The plot twists and turns in ways you would never expect. Truly a masterpiece.",
        "status": "active"
    },
    {
        "_id": 2,
        "user": "Bob",
        "movie": "The Matrix",
        "rating": 4,
        "comment": "The Matrix offers great visuals. The special effects are truly groundbreaking, making it a visual feast.",
        "status": "active"
    },
    {
        "_id": 3,
        "user": "Charlie",
        "movie": "Interstellar",
        "rating": 5,
        "comment": "Interstellar is a mind-blowing experience! The scientific concepts are intriguing and the storyline is deeply moving.",
        "status": "inactive"
    },
    {
        "_id": 4,
        "user": "Diana",
        "movie": "Inception",
        "rating": 4,
        "comment": "Inception is a good movie, but it can be quite confusing. The plot is complex and requires your full attention.",
        "status": "active"
    },
    {
        "_id": 5,
        "user": "Eve",
        "movie": "The Matrix Reloaded",
        "rating": 2,
        "comment": "The Matrix Reloaded didn't quite live up to the first one. It lacked the originality and depth of its predecessor.",
        "status": "inactive"
    },
    {
        "_id": 6,
        "user": "Frank",
        "movie": "Inception",
        "rating": 5,
        "comment": "Inception is a true masterpiece of modern cinema. The storytelling is innovative and the cinematography is stunning.",
        "status": "active"
    },
    {
        "_id": 7,
        "user": "Grace",
        "movie": "Inception",
        "rating": 4,
        "comment": "Inception boasts an intricate plot and stunning visual effects. It's a cinematic journey like no other.",
        "status": "active"
    },
    {
        "_id": 8,
        "user": "Heidi",
        "movie": "The Godfather",
        "rating": 5,
        "comment": "The Godfather is an all-time classic. The storytelling is compelling and the characters are unforgettable.",
        "status": "active"
    },
    {
        "_id": 9,
        "user": "Ivan",
        "movie": "Inception",
        "rating": 4,
        "comment": "Inception is a thrilling ride. It keeps you on the edge of your seat from start to finish.",
        "status": "active"
    },
    {
        "_id": 10,
        "user": "Judy",
        "movie": "Inception",
        "rating": 3,
        "comment": "Inception offers exceptional storytelling and visuals, but the plot can be a bit hard to follow at times.",
        "status": "active"
    },
]

# AST Nodes
class ASTNode:
    pass

class Aggregation(ASTNode):
    def __init__(self, collection, pipeline):
        self.collection = collection
        self.pipeline = pipeline  # List of pipeline stages

# Parser
class Parser:
    def __init__(self, query):
        self.query = query

    def parse(self):
        # Use regex to extract the pipeline string
        match = re.search(r'db\.collection\.aggregate\s*\(\s*(\[[\s\S]*\])\s*\)', self.query)
        if not match:
            raise SyntaxError("Could not find 'aggregate' function with a pipeline")
        pipeline_str = match.group(1)
        print("Collected Pipeline String:", pipeline_str)  # Debug
        try:
            pipeline = json.loads(pipeline_str)
            if not isinstance(pipeline, list):
                raise SyntaxError("Aggregation pipeline should be a list of stages")
            print("Parsed Pipeline:", json.dumps(pipeline, indent=4))  # Debug
        except json.JSONDecodeError as e:
            raise SyntaxError(f"Invalid JSON pipeline: {e}")
        return Aggregation("collection", pipeline)

# Interpreter
class Interpreter:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, node):
        if isinstance(node, Aggregation):
            return self.evaluate_aggregation(node)
        else:
            raise ValueError(f'Unknown node type: {type(node)}')

    def evaluate_aggregation(self, node):
        results = self.dataset.copy()
        print(f"Initial Dataset ({len(results)} documents):")
        for doc in results:
            print(json.dumps(doc, indent=4))
        for idx, stage in enumerate(node.pipeline, start=1):
            if not isinstance(stage, dict) or len(stage) != 1:
                raise ValueError(f"Each pipeline stage must be a single-key dictionary. Invalid stage: {stage}")
            operator, params = next(iter(stage.items()))
            method_name = f'stage_{operator[1:]}'.lower()  # e.g., '$match' -> 'stage_match'
            if not hasattr(self, method_name):
                raise ValueError(f"Unsupported pipeline stage '{operator}'")
            method = getattr(self, method_name)
            print(f"\nApplying Stage {idx}: {operator} with parameters {params}")
            results = method(results, params)
            print(f"Dataset after Stage {idx} ({len(results)} documents):")
            for doc in results:
                print(json.dumps(doc, indent=4))
            # Apply middleware only if the stage is $prompt
            if operator == '$prompt':
                results = self.apply_middleware(results)
                print(f"Dataset after Middleware ({len(results)} documents):")
                for doc in results:
                    print(json.dumps(doc, indent=4))
        return results

    # Stage Implementations
    def stage_match(self, documents, condition):
        """Filter documents based on condition."""
        print("  > $match stage processing...")
        matched = [doc for doc in documents if self.match_filter(doc, condition)]
        print(f"    Matched {len(matched)} documents")
        return matched

    def stage_addfields(self, documents, new_fields):
        """Add or update fields in documents."""
        print("  > $addFields stage processing...")
        for doc in documents:
            for key, expr in new_fields.items():
                if isinstance(expr, dict):
                    # Handle specific expressions, e.g., {"$concat": ["x", "$field_a"]}
                    if '$concat' in expr:
                        parts = expr['$concat']
                        concatenated = ''
                        for part in parts:
                            if isinstance(part, str) and part.startswith('$'):
                                field_name = part[1:]
                                field_value = doc.get(field_name, '')
                                concatenated += str(field_value)
                            else:
                                concatenated += str(part)
                        doc[key] = concatenated
                    else:
                        raise ValueError(f"Unsupported expression in $addFields: {expr}")
                elif isinstance(expr, str) and expr.startswith('$'):
                    field_name = expr[1:]
                    doc[key] = doc.get(field_name, '')
                else:
                    doc[key] = expr
        print("    $addFields stage completed")
        return documents

    def stage_prompt(self, documents, transform):
        """
        Custom $prompt stage that processes the entire list of documents based on a specified field and text.
        Example: {"$prompt": {"comment": "Reviewed: "}}
        This will create a new list of objects with the specified field modified.
        """
        print("  > $prompt stage processing...")
        prompt_field = None
        prompt_text = None

        # Extract 'field' and 'text' from transform
        if len(transform) != 1:
            raise ValueError("$prompt stage expects exactly one field to transform.")
        for field, text in transform.items():
            prompt_field = field
            prompt_text = text

        #print(prompt_field) #comment field
        #print(prompt_text) #prompt to LLM
        
        for doc in documents:
            if prompt_field in doc and isinstance(doc[prompt_field], str):
                response = ollama.chat(model=desiredModel, messages=[
                {
                    'role': 'user',
                    'content': f"""
                    [prompt]
                    {prompt_text}
                    [/prompt]
                    [context]
                        field:{prompt_field}
                        value:
                        {str(doc[prompt_field])}
                        [full document]
                        {str(doc)}
                        [/full document]
                    [/context]
        """,
                },
                ])
                doc['prompt_output'] = response['message']['content']
                print(f"    Created new object for: {doc}")
            else:
                print(f"    Skipping document ID {doc.get('_id')} as it lacks the field '{prompt_field}' or it's not a string.")

        print("    $prompt stage completed")
        return documents

    # Middleware Implementation
    def apply_middleware(self, documents):
        """
        Middleware function that processes each document after the $prompt stage.
        You can customize this function to perform actions such as logging, modifying documents, etc.
        """
        print("  > Middleware processing each document...")
        for doc in documents:
            # Example middleware action: Log the prompted_comments
            prompted_comments = doc.get('prompted_comments', [])
            print(f"    Document ID {doc['_id']} has prompted comments:")
            for pc in prompted_comments:
                print(f"      - User: {pc['user']}, Movie: {pc['movie']}, Comment: {pc['comment']}")
            # Example: Add a new field 'processed' set to True
            doc['processed'] = True
        print("  > Middleware processing completed")
        return documents

    # Helper Methods
    def match_filter(self, document, filter_expr):
        for key, value in filter_expr.items():
            if not self.match_condition(document, key, value):
                return False
        return True

    def match_condition(self, document, key, value):
        if key.startswith('$'):
            if key == '$and':
                if not isinstance(value, list):
                    raise ValueError(f"$and operator expects a list, got {type(value)}")
                return all(self.match_filter(document, cond) for cond in value)
            elif key == '$or':
                if not isinstance(value, list):
                    raise ValueError(f"$or operator expects a list, got {type(value)}")
                return any(self.match_filter(document, cond) for cond in value)
            elif key == '$not':
                return not self.match_filter(document, value)
            else:
                raise ValueError(f"Unsupported operator '{key}'")
        else:
            doc_value = document.get(key)
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if not self.evaluate_operator(doc_value, op, op_value):
                        return False
                return True
            else:
                return doc_value == value

    def evaluate_operator(self, doc_value, operator, value):
        if operator == '$gt':
            return doc_value > value
        elif operator == '$lt':
            return doc_value < value
        elif operator == '$gte':
            return doc_value >= value
        elif operator == '$lte':
            return doc_value <= value
        elif operator == '$eq':
            return doc_value == value
        elif operator == '$ne':
            return doc_value != value
        else:
            raise ValueError(f"Unsupported operator '{operator}'")

# Main function
def main():
    # Aggregation pipeline with $match and custom $prompt stages
    query = '''
    db.collection.aggregate([
        {"$match": {"rating": {"$gt": 3}, "movie": "Inception"}},
        {"$prompt": {"comment": "Summarize the comment in 5 words"}}
    ])
    '''
    
    print(f"Executing Query: {query.strip()}\n")
    
    parser = Parser(query)
    try:
        ast = parser.parse()
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return
    
    interpreter = Interpreter(DATASET)
    try:
        results = interpreter.evaluate(ast)
    except ValueError as e:
        print(f"Runtime error: {e}")
        return
    
    print("\nFinal Query Results:")
    if results:
        for doc in results:
            print(json.dumps(doc, indent=4))
    else:
        print("No documents matched the query.")

if __name__ == '__main__':
    main()
