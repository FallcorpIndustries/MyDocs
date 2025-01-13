from app import app, db

with app.app_context():
    db.drop_all()
    print("All tables dropped.")

    db.create_all()
