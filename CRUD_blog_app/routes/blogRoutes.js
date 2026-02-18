const express = require('express');
const router = express.Router();

// 1. IMPORT FIRST - This makes sure the functions exist before we use them
const {
    blog_index,
    blog_details,
    blog_create_get,
    blog_create_post,
    blog_delete,
    blog_update_get,
    blog_update_post
} = require('../controllers/blogController');

// 2. DEFINE ROUTES SECOND
router.get('/create', blog_create_get);
router.get('/', blog_index);
router.post('/', blog_create_post);
router.get('/:id', blog_details);
router.get('/update/:id', blog_update_get);
router.post('/update/:id', blog_update_post);
router.delete('/:id', blog_delete);

module.exports = router;