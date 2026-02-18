const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const studentSchema = new Schema({
    "Marital status": Number,
    "Course": Number,
    "Age at enrollment": Number,
    "student_id": String, // this for the ID on student cards
    "risk_score": Number  // what the Python AI provides
}, { timestamps: true });

const Student = mongoose.model('Student', studentSchema, 'students');
module.exports = Student;