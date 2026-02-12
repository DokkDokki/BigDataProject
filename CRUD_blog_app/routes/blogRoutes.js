const express = require('express');
const blogController = require('../controllers/blogController');
const router = express.Router();

router.get('/create', blogController.blog_create_get);
router.get('/', blogController.blog_index);
router.post('/', blogController.blog_create_post);
router.get('/:id', blogController.blog_details);
router.get('/update/:id', blogController.blog_update_get);
router.post('/update/:id', blogController.blog_update_post);
router.delete('/:id', blogController.blog_delete);

module.exports = router;
