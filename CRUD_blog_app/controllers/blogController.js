const Student = require('../models/blog'); 
const axios = require('axios');

// 1. Display all students on the dashboard
const blog_index = (req, res) => {
    Student.find().sort({ createdAt: -1 })
        .then(result => {
            res.render('index', { blogs: result, title: 'All Students' });
        })
        .catch(err => console.log(err));
};


const blog_create_get = (req, res) => {
    res.render('create', { title: 'Add New Student' });
};

const blog_create_post = async (req, res) => {
    try {
        const studentData = {
            "Marital status": parseInt(req.body.marital_status),
            "Course": parseInt(req.body.course),
            "Age at enrollment": parseInt(req.body.age),
            "Curricular units 1st sem (approved)": parseInt(req.body.units_approved),
            "Curricular units 1st sem (grade)": parseFloat(req.body.units_grade)
        };
        const pythonResponse = await axios.post('http://127.0.0.1:8000/get_risk', studentData);
        const student = new Student({
            ...studentData,
            student_id: req.body.student_id,
            risk_score: pythonResponse.data.risk_score
        });
        await student.save();
        res.redirect('/blogs');
    } catch (err) {
        console.log(err);
        res.status(500).send("AI calculation failed.");
    }
};

const blog_details = (req, res) => {
    const id = req.params.id;
    Student.findById(id)
        .then(result => res.render('details', { blog: result, title: 'Student Details' }))
        .catch(err => res.status(404).render('404', { title: 'Not found' }));
};

const blog_delete = (req, res) => {
    const id = req.params.id;
    Student.findByIdAndDelete(id)
        .then(result => res.json({ redirect: '/blogs' }))
        .catch(err => console.log(err));
};

// For the Update routes
const blog_update_get = (req, res) => {
    const id = req.params.id;
    Student.findById(id)
        .then(result => res.render('update', { blog: result, title: 'Update Student' }))
        .catch(err => res.status(404).render('404', { title: 'Not found' }));
};

const blog_update_post = async (req, res) => {
    const id = req.params.id;
    // ... logic to update and re-run Python AI ...
};

// Export everything so blogRoutes can see them
module.exports = {
    blog_index,
    blog_details,
    blog_create_get,
    blog_create_post,
    blog_delete,
    blog_update_get,
    blog_update_post
};