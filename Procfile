web: uvicorn src.api.server:app --host 0.0.0.0 --port $PORT
dashboard: streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
