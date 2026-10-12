from flask import Flask, render_template

from config import SECRET_KEY
from routes.prediction import prediction_bp
from database.models import get_all_predictions
from routes.reports import reports_bp
from routes.dashboard import dashboard_bp

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY

    app.register_blueprint(prediction_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(dashboard_bp)
    
    @app.route("/")
    def home():
        return """
        <h1>AI Predictive Maintenance System</h1>
        <p>Application is running successfully.</p>
        <a href="/predict/">Go to Prediction</a>
        """

    @app.route("/history")
    def history():
        predictions = get_all_predictions()

        return render_template(
            "history.html",
            predictions=predictions
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)