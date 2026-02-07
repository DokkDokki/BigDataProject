1. create your own .env file

Copy from .env.example
MONGO_URI=mongodb+srv://<USERNAME>:<PASSWORD>@cluster0.hagjvo3.mongodb.net/student_dropout_db?retryWrites=true&w=majority&authSource=admin

2. Test connection by running:

python load_data.py

Expected:
Ping OK
Collections: ['students']


