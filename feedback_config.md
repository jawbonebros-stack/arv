# Feedback System Configuration Guide

The Onsenses feedback system can be configured in multiple ways to customize its behavior, appearance, and data collection.

## 1. Button Configuration

**File:** `templates/components/feedback_modal.html` (CSS section, lines 168-216)

### Button Position
```css
.feedback-float-btn {
    bottom: 30px;    /* Distance from bottom (default: 30px) */
    right: 30px;     /* Distance from right (default: 30px) */
    width: 56px;     /* Button size (default: 56px) */
    height: 56px;
}
```

### Button Appearance
```css
.feedback-float-btn {
    background: #6c757d;     /* Button color (default: neutral gray) */
    color: white;            /* Icon color */
    border-radius: 50%;      /* Shape (50% = circle) */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);  /* Shadow */
}
```

### Button Icon
Change the icon in the HTML section:
```html
<div id="feedbackButton" class="feedback-float-btn" title="Share Feedback">
    💬  <!-- Change this emoji or use <i data-feather="message-circle"></i> -->
</div>
```

## 2. Workflow Detection Configuration

**File:** `templates/components/feedback_modal.html` (lines 311-364)

### Add New Workflow Types
```javascript
function detectCurrentWorkflow() {
    const path = window.location.pathname;
    
    // Add your custom workflow detection here
    if (path.includes('/your-new-page')) {
        return {
            name: 'your_workflow_name',
            title: 'Your Workflow Title',
            description: 'Description shown to users',
            context: 'your_context'
        };
    }
    // ... existing workflows ...
}
```

### Modify Existing Workflows
Edit the return objects to change titles, descriptions, or detection logic:
```javascript
if (path.includes('/wizard')) {
    return {
        name: 'trial_creation',           // Database storage name
        title: 'Trial Creation Wizard',   // Shown to users
        description: 'Creating a new prediction trial',  // Modal description
        context: 'trial_wizard'          // Context for filtering
    };
}
```

## 3. Form Configuration

**File:** `templates/components/feedback_modal.html` (lines 1-167)

### Rating Categories
Modify or add rating questions:
```html
<!-- Overall Experience Rating -->
<div class="rating-group">
    <label class="form-label">Overall Experience</label>
    <div class="rating-stars" data-rating="0">
        <!-- 5 star rating system -->
    </div>
    <input type="hidden" name="overall_rating" value="">
</div>
```

### Issue Checkboxes
Add/modify common issues:
```html
<div class="form-check">
    <input class="form-check-input" type="checkbox" name="issues" value="your_new_issue">
    <label class="form-check-label">Your New Issue Description</label>
</div>
```

### Text Fields
Modify feedback text areas:
```html
<div class="mb-3">
    <label for="feedbackText" class="form-label">Your Feedback</label>
    <textarea class="form-control" id="feedbackText" name="feedback_text" 
              rows="3" placeholder="Tell us about your experience..."></textarea>
</div>
```

## 4. Backend Configuration

**File:** `main.py` (feedback API endpoint)

### Database Storage
The feedback is stored in the `UserFeedback` table with these fields:
- `workflow` - The workflow name (from JavaScript detection)
- `context` - Additional context information
- `overall_rating` - 1-5 star rating
- `ease_rating` - 1-5 star rating for ease of use
- `feedback_text` - User's text feedback
- `suggestions` - User suggestions
- `issues` - JSON array of selected issues
- `contact_ok` - Whether user agrees to be contacted
- `page_url` - URL where feedback was submitted
- `user_agent` - Browser information

### API Endpoint Customization
Modify the `/api/feedback` endpoint in `main.py` to:
- Add validation rules
- Modify storage logic
- Add email notifications
- Integrate with external services

## 5. Admin Dashboard Configuration

**File:** `templates/admin/feedback_dashboard.html`

### Statistics Display
Modify the analytics shown:
```html
<div class="col-md-3">
    <div class="card stats-card text-center">
        <div class="card-body">
            <h4 class="text-secondary">{{ stats.total_feedback }}</h4>
            <h6 class="text-muted">Total Feedback</h6>
        </div>
    </div>
</div>
```

### Filtering Options
Add new filter categories in the filter form:
```html
<select class="form-select" name="workflow">
    <option value="">All Workflows</option>
    <option value="trial_creation">Trial Creation</option>
    <!-- Add your new workflow options here -->
</select>
```

## 6. Automatic Trigger Configuration

**File:** Various template files

### Timing-Based Triggers
Add automatic feedback prompts after certain actions:
```javascript
// Show feedback after successful trial creation
setTimeout(() => {
    triggerWorkflowFeedback('trial_creation', 'completed_creation', 3000);
}, 1000);
```

### Event-Based Triggers
Trigger feedback based on user actions:
```javascript
// Trigger after completing multiple ARV outcomes
if (outcomeCompletionCount >= 2) {
    setTimeout(() => {
        triggerWorkflowFeedback('arv_session', 'completed_multiple_outcomes', 5000);
    }, 2000);
}
```

## 7. Styling Configuration

**File:** `templates/components/feedback_modal.html` (CSS section)

### Modal Appearance
```css
.modal-content {
    border-radius: 15px;     /* Rounded corners */
    border: none;            /* Remove border */
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);  /* Shadow */
}
```

### Star Rating Colors
```css
.star.filled {
    color: #ffc107;          /* Filled star color */
}

.star.temp-fill {
    color: #ffdc7a;          /* Hover color */
}
```

## 8. Enable/Disable Configuration

### Hide Feedback Button on Specific Pages
Add to the page template:
```html
<style>
.feedback-float-btn { display: none !important; }
</style>
```

### Conditional Display
Modify the JavaScript to show/hide based on conditions:
```javascript
// Only show feedback button for logged-in users
if (user_is_logged_in) {
    document.getElementById('feedbackButton').style.display = 'flex';
}
```

## Quick Configuration Examples

### 1. Change Button Color to Blue
```css
.feedback-float-btn {
    background: #007bff;  /* Bootstrap blue */
}
```

### 2. Move Button to Left Side
```css
.feedback-float-btn {
    left: 30px;   /* Instead of right: 30px */
}
```

### 3. Add Custom Workflow Detection
```javascript
if (path.includes('/my-custom-page')) {
    return {
        name: 'custom_workflow',
        title: 'Custom Process',
        description: 'Working on custom feature',
        context: 'custom'
    };
}
```

### 4. Add New Issue Type
```html
<div class="form-check">
    <input class="form-check-input" type="checkbox" name="issues" value="performance_slow">
    <label class="form-check-label">Page loads too slowly</label>
</div>
```

All configuration changes take effect immediately when you save the files - no server restart required for frontend changes.