import os

import requests
from flask import Flask, redirect, render_template, request, url_for


app = Flask(__name__)

BACKEND_URL = os.environ.get(
    "BACKEND_URL", "http://backend:8000"
).rstrip("/")

TASK_STATUSES = ["Offen", "In Bearbeitung", "Erledigt"]


@app.get("/")
def dashboard():
    try:
        response = requests.get(
            f"{BACKEND_URL}/projects",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError):
        return (
            "The backend is currently unavailable. "
            "Please refresh in a moment.",
            503,
        )

    return render_template(
        "dashboard.html",
        projects=data["projects"],
        phases=data["phases"],
        task_statuses=TASK_STATUSES,
    )


@app.post("/tasks/<int:task_id>/status")
def update_task_status(task_id):
    status = request.form.get("status")

    if status not in TASK_STATUSES:
        return "Invalid task status.", 400

    try:
        response = requests.patch(
            f"{BACKEND_URL}/tasks/{task_id}",
            json={"status": status},
            timeout=10,
        )
        response.raise_for_status()

    except requests.RequestException:
        return (
            "The task could not be updated. Please try again.",
            502,
        )

    return redirect(url_for("dashboard"), code=303)