web: /opt/venv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port $PORT
dashboard: /opt/venv/bin/streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
