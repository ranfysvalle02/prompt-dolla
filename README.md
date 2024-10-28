# prompt-dolla 

---

# A new frontier: Embedding Language Models into Database Workflows

*Imagine if your database could not only store and retrieve data but also understand it, summarize it, and provide insights—all within your existing query workflows. Sounds exciting, right? Let's explore how integrating language models into databases is turning this vision into reality.*

---

## Introduction

Databases have always been the workhorses behind our applications, efficiently storing and retrieving data. But as the volume and complexity of data grow, so does the need for smarter ways to handle it. What if we could make databases more intelligent, enabling them to interpret and process data in ways we never thought possible?

In this post, we'll delve into how embedding language models into database workflows can revolutionize data handling. We'll walk through a Python script that integrates a language model into a MongoDB-like aggregation pipeline, enhancing data processing and opening up a world of new possibilities. Whether you're a developer, data scientist, or just curious about the future of data, this journey will spark your imagination.

Integrating language models into databases requires a robust and flexible data platform, and MongoDB fits this role perfectly. It offers more than just a traditional database—it provides a comprehensive platform ideal for AI-driven applications.

---

## Table of Contents

1. [The Evolution of Databases](#the-evolution-of-databases)
2. [Introducing Language Models into Queries](#introducing-language-models-into-queries)
3. [Understanding the Dataset](#understanding-the-dataset)
4. [Building the Custom Query Interpreter](#building-the-custom-query-interpreter)
5. [Enhancing Queries with the `$prompt` Stage](#enhancing-queries-with-the-prompt-stage)
6. [Real-World Applications](#real-world-applications)
7. [Envisioning the Future](#envisioning-the-future)
8. [Conclusion](#conclusion)

---

## The Evolution of Databases

Traditional databases excel at storing and retrieving structured data quickly and reliably. They're the backbone of countless applications, from e-commerce platforms to social networks. However, as data becomes more unstructured and complex—think user reviews, social media posts, or customer feedback—the limitations of traditional databases become apparent.

MongoDB's document-oriented model stores data in a flexible, JSON-like format, accommodating unstructured and semi-structured data common in AI tasks. This flexibility allows for easy adaptation as your data evolves, without the constraints of rigid schemas.

### Challenges with Unstructured Data

- **Limited Understanding**: Databases store text but don't understand its meaning.
- **Complex Queries**: Extracting insights from unstructured data requires complex queries and additional processing.
- **Separate Systems**: Often, data needs to be exported to external tools for analysis, adding complexity and delay.

### MongoDB: Powerful Querying and Aggregation

MongoDB excels in advanced querying capabilities essential for AI:

- **Text Search**: Perform sophisticated text searches, including phrases and relevance scoring, directly within the database.
- **Keyword Search and Filtering**: Efficiently index and retrieve specific keywords from large datasets.
- **Time-Series Data Handling**: Natively support and analyze time-stamped data, crucial for logs and sensor data.
- **Complex Aggregations**: Use the aggregation framework to process and transform data in-place, eliminating the need for external processing tools. The aggregation framework saves hours of work, and lines of code.
  
### The Promise of Language Models

Language models like GPT-4 have shown remarkable abilities to understand and generate human-like text. By integrating these models directly into database workflows, we can:

- **Interpret Unstructured Data**: Summarize, analyze, or classify text data within the database.
- **Generate Insights**: Produce insights or recommendations on the fly.
- **Simplify Workflows**: Eliminate the need for external data processing tools.

---

## Introducing Language Models into Queries

So, how do we bring these two worlds together? Let's explore a Python script that demonstrates this integration using a custom query interpreter and a language model.

### The Concept

We'll simulate a MongoDB-like aggregation pipeline but introduce a custom `$prompt` stage. This stage leverages a language model to process data within the query workflow.

### The Tools

- **Python**: For scripting and building the interpreter.
- **Ollama**: A library to interact with language models.
- **A Sample Dataset**: A collection of movie reviews with user comments.

---

## Understanding the Dataset

Our dataset consists of user comments on movies:

```python
DATASET = [
    {
        "_id": 1,
        "user": "Alice",
        "movie": "Inception",
        "rating": 5,
        "comment": "This movie is absolutely amazing! The plot twists and turns in ways you would never expect. Truly a masterpiece.",
        "status": "active"
    },
    # ... additional documents ...
]
```

We have ten documents, each representing a user's comment on a movie, along with their rating and status.

---

## Building the Custom Query Interpreter

Our script mimics a MongoDB aggregation pipeline, with stages that process data step by step.

### The Components

- **Parser**: Extracts the pipeline stages from the query string.
- **Interpreter**: Evaluates each stage of the pipeline against the dataset.
- **AST Nodes**: Represents the structure of the query.

### The Query

Here's the query we'll be working with:

```python
query = '''
db.collection.aggregate([
    {"$match": {"rating": {"$gt": 3}, "movie": "Inception"}},
    {"$prompt": {"comment": "Summarize the comment in 5 words"}}
])
'''
```

#### What's Happening Here?

1. **`$match` Stage**: Filters documents where the rating is greater than 3 and the movie is "Inception".
2. **`$prompt` Stage**: Uses the language model to summarize the `comment` field in each document.

### Parsing the Query

The `Parser` class uses regular expressions to extract the pipeline:

```python
class Parser:
    def parse(self):
        match = re.search(r'db\.collection\.aggregate\s*\(\s*(\[[\s\S]*\])\s*\)', self.query)
        pipeline_str = match.group(1)
        pipeline = json.loads(pipeline_str)
        return Aggregation("collection", pipeline)
```

### Evaluating the Pipeline

The `Interpreter` class processes each stage:

```python
class Interpreter:
    def evaluate(self, node):
        if isinstance(node, Aggregation):
            return self.evaluate_aggregation(node)
```

---

## Enhancing Queries with the `$prompt` Stage

The real magic happens in the custom `$prompt` stage, where we integrate the language model.

### How the `$prompt` Stage Works

```python
def stage_prompt(self, documents, transform):
    # Extract 'field' and 'text' from transform
    prompt_field, prompt_text = next(iter(transform.items()))
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
```

#### Breaking It Down

- **Extracts the Target Field**: The field to process (e.g., `comment`).
- **Sends a Prompt to the Language Model**: Instructs the model to perform an action (e.g., "Summarize the comment in 5 words").
- **Stores the Response**: Adds a new field `prompt_output` to the document with the model's output.

### Middleware Processing

Optionally, a middleware function can process documents after the `$prompt` stage:

```python
def apply_middleware(self, documents):
    for doc in documents:
        # Example: Mark document as processed
        doc['processed'] = True
    return documents
```

---

## Real-World Applications

Integrating language models into databases isn't just a neat trick—it's a transformative approach with broad implications. Let's explore some real-world scenarios.

### 1. Customer Feedback Analysis

**Scenario**: A company wants to quickly understand customer sentiments from reviews or support tickets.

**Solution**:

- **Data**: Customer reviews stored in a database.
- **Query**: Use a `$prompt` stage to summarize each review or extract sentiments.
- **Benefit**: Faster insights into customer opinions, leading to improved products and services.

### 2. Content Moderation

**Scenario**: A social media platform needs to moderate user-generated content efficiently.

**Solution**:

- **Data**: Posts and comments in the database.
- **Query**: Apply the language model to flag inappropriate content or summarize lengthy posts.
- **Benefit**: Keeps the community safe while reducing moderation workload.

### 3. Automated Report Generation

**Scenario**: A financial firm wants to generate summaries of complex financial documents.

**Solution**:

- **Data**: Financial reports stored as documents.
- **Query**: Use the language model to extract key insights or summarize reports.
- **Benefit**: Saves analysts' time, allowing them to focus on decision-making.

### 4. Personalized Learning

**Scenario**: An educational platform aims to provide personalized summaries of learning materials.

**Solution**:

- **Data**: Educational content in the database.
- **Query**: Use the language model to generate summaries tailored to individual learning levels.
- **Benefit**: Enhances the learning experience by catering to individual needs.

### 5. Healthcare Data Interpretation

**Scenario**: Medical professionals need quick summaries of patient records.

**Solution**:

- **Data**: Patient records in a secure database.
- **Query**: Use the language model to summarize medical histories or highlight critical information.
- **Benefit**: Improves patient care through rapid access to vital data.

---
### Pushing the Boundaries: Future Possibilities

#### Hyper-Personalized E-Commerce Experiences

Imagine e-commerce platforms that generate personalized product descriptions and recommendations in real-time. As a user browses, the system analyzes their behavior, preferences, and even past interactions. Embedded language models could then craft product descriptions that highlight features most relevant to that user, making the shopping experience more engaging and increasing the likelihood of a purchase.

#### Intelligent Data Cleaning and Enrichment

Data quality is crucial for accurate analysis. Language models could automatically detect inconsistencies, correct errors, and fill in missing information within your datasets. They could also enrich data by extracting entities and relationships, providing a deeper understanding of the information you already have—all handled seamlessly within the database.

#### Automated Contract Summarization

Embedded language models could analyze lengthy legal documents stored in the database, providing concise summaries and highlighting critical clauses or potential issues.

#### Predictive Analysis 

Language models interpret patient notes, lab results, and symptom descriptions to predict potential health issues.

---

## Conclusion

In the era of data-driven innovation, MongoDB eliminates barriers between your data and actionable insights. Its versatile platform is ideally suited for integrating language models, enabling you to build intelligent applications with ease. By choosing MongoDB, you're not just selecting a database—you're investing in a platform that empowers you to harness the full potential of AI.

We're on the brink of a new era where databases don't just store data—they understand it. By embedding language models into database workflows, we unlock unprecedented capabilities.

---

**What's Next?**

As technology continues to advance, we can expect:

- **More Powerful Models**: Improved language models offering deeper understanding and faster responses.
- **Seamless Integration**: Databases and AI models working together out of the box.
- **Innovative Applications**: New use cases across industries, from finance to healthcare to education.

## Understanding the Risks of Integrating Language Models

While the integration of language models into database workflows unlocks exciting possibilities, it's crucial to consider the associated risks to ensure responsible and effective implementation. Here are some key risks to be aware of:

### Data Privacy and Security

**Risk**: Incorporating language models may involve processing sensitive data, which could be exposed if not handled properly. If the language model is hosted externally or relies on third-party services, there's a risk of data breaches or unauthorized access.

**Mitigation**:

- **Data Anonymization**: Remove or mask personally identifiable information (PII) before processing data with the language model.
- **On-Premises Models**: Use locally hosted models to keep all data within your organization's secure environment.
- **Encryption**: Implement strong encryption protocols for data in transit and at rest.

### Model Bias and Ethical Considerations

**Risk**: Language models are trained on large datasets that may contain biases. This can lead to biased outputs, perpetuating stereotypes or unfair practices.

**Mitigation**:

- **Bias Evaluation**: Regularly assess the model's outputs for biased or unethical content.
- **Model Fine-Tuning**: Retrain or fine-tune models on curated datasets that minimize bias.
- **Human Oversight**: Include human review processes for critical tasks to catch and correct biased outputs.

### Accuracy and Reliability

**Risk**: Language models can produce incorrect or nonsensical results, especially when dealing with ambiguous or complex queries. Relying solely on model outputs can lead to misinformation or flawed decision-making.

**Mitigation**:

- **Result Validation**: Implement checks to verify the accuracy of the model's outputs.
- **Confidence Thresholds**: Use confidence scoring to determine when to trust the model's results or flag them for review.
- **Fallback Mechanisms**: Provide alternative processing methods if the model's output doesn't meet reliability standards.

### Performance and Scalability

**Risk**: Language models, particularly large ones, require significant computational resources. Integrating them into database workflows can impact performance, leading to increased latency or resource bottlenecks.

**Mitigation**:

- **Resource Planning**: Assess and provision the necessary computational resources to handle the additional load.
- **Model Optimization**: Use smaller, optimized models or implement techniques like model distillation to reduce resource consumption.
- **Asynchronous Processing**: Offload language model tasks to background processes to minimize impact on real-time operations.

### Compliance and Regulatory Challenges

**Risk**: Processing data with language models may conflict with data protection regulations like GDPR, HIPAA, or other industry-specific compliance standards.

**Mitigation**:

- **Regulatory Compliance Review**: Consult with legal experts to ensure that data processing practices comply with relevant regulations.
- **Data Residency**: Ensure that data remains within required geographic boundaries if regulations mandate it.
- **Audit Trails**: Maintain detailed logs of data processing activities for accountability and compliance audits.

### Operational Complexity

**Risk**: Introducing language models adds complexity to your systems, requiring specialized knowledge for maintenance, updates, and troubleshooting.

**Mitigation**:

- **Team Training**: Invest in training your team on AI and language model technologies.
- **Documentation**: Maintain thorough documentation of how language models are integrated into your workflows.
- **Monitoring and Alerting**: Implement monitoring tools to track the performance and health of the language model components.

### Dependency on External Services

**Risk**: Relying on third-party language models or APIs can introduce dependencies that are outside your control, such as service outages or changes in service terms.

**Mitigation**:

- **Service Level Agreements (SLAs)**: Establish clear SLAs with service providers to ensure reliability.
- **Redundancy**: Have fallback options or alternative providers in case of service disruptions.
- **Self-Hosted Options**: Consider using open-source models that can be hosted internally to reduce external dependencies.

### Ethical Use and User Trust

**Risk**: Users may be unaware that AI is processing their data, leading to trust issues, especially if outputs are unexpected or if data is used without explicit consent.

**Mitigation**:

- **Transparency**: Inform users about how their data is being processed and the role of AI in your systems.
- **Consent Mechanisms**: Obtain explicit consent from users where necessary.
- **Responsible AI Policies**: Develop and adhere to ethical guidelines for AI use within your organization.

---

## FULL CODE

```python
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
```

---
