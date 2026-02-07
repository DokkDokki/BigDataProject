## MongoDB Setup (Important)

- This project uses a shared MongoDB Atlas cluster.
- Data is already imported into database: `student_dropout_db`
- Collection name: `students`

### Steps for team members:
1. Ask project owner to add you as a MongoDB Database User
2. Create your own `.env` file from `.env.example`
3. Put your own MongoDB connection string in `MONGO_URI`
4. Run:
   ```bash
   python load_data.py



