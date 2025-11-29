def create_web_app():
    """Flask/FastAPI web application wrapper for production use"""
    # Example structure for a REST API
    # from flask import Flask, request, jsonify
    # 
    # app = Flask(__name__)
    # rag = RAGPipeline()
    # 
    # @app.route('/ingest', methods=['POST'])
    # def ingest():
    #     data = request.json
    #     chunks = rag.add_document(data['text'], data.get('metadata'))
    #     return jsonify({'chunks': chunks})
    # 
    # @app.route('/query', methods=['POST'])
    # def query():
    #     data = request.json
    #     result = rag.query(data['question'])
    #     return jsonify(result)
    # 
    # if __name__ == '__main__':
    #     app.run(debug=True, port=5000)
    pass


if __name__ == "__main__":
    # Run the basic demo by default
    run_basic_demo()
    
    # Uncomment to run advanced demo
    # run_advanced_demo()