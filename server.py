"""Entry point: python server.py starts Loan Desk at http://localhost:8000.

Kept as a thin script on purpose - all the real logic lives in loan_desk/.
"""
from loan_desk import config, http_server, model_service

if __name__ == "__main__":
    if not config.KEY:
        raise SystemExit("Put LLM_API_KEY=... in the .env file")
    model_service.log_feature_mapping()
    print(f"Open http://localhost:8000  (chat model: {config.MODEL}, "
          f"PD model: {model_service.B['metrics']['model']}, test AUC {model_service.B['metrics']['auc']})")
    http_server.serve()
