import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Literal

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import logging
from rag import answer_question


PHASES = [
    "Anfrage",
    "Angebote",
    "Aufträge",
    "Montagebereit",
    "Terminiert",
    "Rechnung",
    "Controlling",
]


def connect():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return psycopg.connect(
            database_url,
            row_factory=dict_row,
        )

    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        row_factory=dict_row,
    )


def initialize_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                customer TEXT NOT NULL,
                manager TEXT NOT NULL,
                phase TEXT NOT NULL,
                deadline DATE NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                project_id INTEGER NOT NULL
                    REFERENCES projects(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                owner TEXT NOT NULL,
                due_date DATE NOT NULL,
                status TEXT NOT NULL DEFAULT 'Offen'
                    CHECK (status IN (
                        'Offen', 'In Bearbeitung', 'Erledigt'
                    ))
            )
        """)

        count = conn.execute(
            "SELECT COUNT(*) AS total FROM projects"
        ).fetchone()["total"]

        # Add sample data only when the database is empty.
        if count == 0:
            today = date.today()

            examples = [
                (
                    "Fenstertausch", "Kunde A", "Mitarbeiter A",
                    "Aufträge", 14,
                    "Fertigungsmaß aufnehmen", "Mitarbeiter B", -2,
                ),
                (
                    "Badmodernisierung", "Kunde B", "Mitarbeiter B",
                    "Angebote", 21,
                    "Angebot nachfassen", "Mitarbeiter A", -1,
                ),
                (
                    "Küchenmodernisierung", "Kunde C", "Mitarbeiter A",
                    "Montagebereit", 10,
                    "Montagetermin abstimmen", "Mitarbeiter C", 2,
                ),
                (
                    "Dachfenster", "Kunde D", "Mitarbeiter C",
                    "Terminiert", 7,
                    "Montage durchführen", "Mitarbeiter B", 7,
                ),
                (
                    "Innenausbau", "Kunde E", "Mitarbeiter B",
                    "Rechnung", 5,
                    "Schlussrechnung vorbereiten", "Mitarbeiter A", 1,
                ),
            ]

            for example in examples:
                (
                    name, customer, manager, phase, project_days,
                    title, owner, task_days,
                ) = example

                project = conn.execute("""
                    INSERT INTO projects
                        (name, customer, manager, phase, deadline)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    name,
                    customer,
                    manager,
                    phase,
                    today + timedelta(days=project_days),
                )).fetchone()

                conn.execute("""
                    INSERT INTO tasks
                        (project_id, title, owner, due_date)
                    VALUES (%s, %s, %s, %s)
                """, (
                    project["id"],
                    title,
                    owner,
                    today + timedelta(days=task_days),
                ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Hanshans Demo API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/projects")
def get_projects():
    with connect() as conn:
        projects = conn.execute(
            "SELECT * FROM projects ORDER BY deadline, id"
        ).fetchall()

        tasks = conn.execute(
            "SELECT * FROM tasks ORDER BY due_date, id"
        ).fetchall()

    today = date.today()
    tasks_by_project = {}

    for task in tasks:
        task["overdue"] = (
            task["due_date"] < today
            and task["status"] != "Erledigt"
        )
        tasks_by_project.setdefault(
            task["project_id"], []
        ).append(task)

    for project in projects:
        project["tasks"] = tasks_by_project.get(
            project["id"], []
        )
        project["overdue_tasks"] = sum(
            task["overdue"] for task in project["tasks"]
        )

    return {"phases": PHASES, "projects": projects}


class TaskUpdate(BaseModel):
    status: Literal["Offen", "In Bearbeitung", "Erledigt"]


@app.patch("/tasks/{task_id}")
def update_task(task_id: int, update: TaskUpdate):
    with connect() as conn:
        task = conn.execute("""
            UPDATE tasks
            SET status = %s
            WHERE id = %s
            RETURNING *
        """, (update.status, task_id)).fetchone()

        if task is None:
            raise HTTPException(
                status_code=404,
                detail="Task not found",
            )

    return task


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(body: QuestionRequest):
    try:
        return answer_question(body.question)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logging.exception("RAG request failed")
        raise HTTPException(
            status_code=503,
            detail="The AI service is currently unavailable.",
        )