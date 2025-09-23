# ARVLab Platform

## Overview
ARVLab is an experimental platform for collaborative prediction research utilizing Associative Remote Viewing (ARV) techniques. It provides a complete prediction trial system with commit-reveal mechanics, real-time collaboration, and AI-enhanced analysis. The platform, built as a FastAPI web application with WebSocket support, facilitates structured prediction experiments across various domains, including stocks, lottery, and sports, with support for binary and ternary outcomes. Key capabilities include collaborative descriptor development, flexible judging workflows, and result aggregation, with optional AI integrations (OpenAI, CLIP) for advanced image analysis. The project aims to provide a comprehensive framework for conducting rigorous ARV studies.

## User Preferences
Preferred communication style: Simple, everyday language.
Service Name: ARVLab
Leaderboard Style: Reddit-style usernames

## Configuration Documentation
- **Feedback System Config**: See `feedback_config.md` for comprehensive customization guide covering button appearance, workflow detection, form fields, admin dashboard, and automatic triggers

## System Architecture

### Recent Changes (January 2025)
- **Lucide Icon System Migration (September 18, 2025)**: Successfully replaced problematic Feather icons library with actively maintained Lucide library, eliminating all JavaScript console errors and ensuring stable icon rendering. Implemented seamless backward compatibility system that automatically converts data-feather attributes to data-lucide, maintaining full functionality while modernizing the icon system. ARV workflows, descriptor submissions, and all user interface elements now operate without JavaScript crashes.
- **7-Digit Target Numbers (August 26, 2025)**: Added unique 7-digit target numbers to all tasks for easy identification and reference. Each task now displays its target number (e.g., "8058556") prominently in the task list, detail pages, and ARV wizard. Numbers are automatically generated and guaranteed unique across the platform, enhancing user experience and task organization.
- **AI Analysis System Implementation (August 17, 2025)**: Successfully implemented comprehensive AI analysis feature with viewing session analysis engine, detailed percentage matching across 6 sensory categories, analysis results page with visual breakdowns and social sharing, "Analyze My Results" button integration, database schema for storing analysis results, and full workflow integration for completed tasks. Fixed database compatibility issues and user attribute references for seamless operation.
- **Error Resolution (August 17, 2025)**: Fixed critical "Error saving descriptors" issue by correcting user attribute references (user.first_name/last_name to user.name) and JavaScript form ID mismatch in ARV wizard. System now successfully saves descriptors across all 6 sensory categories with visual feedback (green checkmark, "Saved!" confirmation). ARV wizard "Save Target Impressions" button fully functional.
- **Group Task Sharing Encouragement (August 16, 2025)**: Added comprehensive sharing page after group task creation with copy-to-clipboard functionality, social media sharing buttons, earning potential emphasis (1 credit per participant), consensus accuracy benefits, and sharing tips to maximize participation and task effectiveness
- **Single Target ARV Methodology (August 15, 2025)**: Updated ARV system to follow proper remote viewing methodology where non-lottery tasks (stocks, sports, binary outcomes) only collect impressions for a single randomly selected target. This prevents displacement to wrong targets and maintains ARV scientific integrity. Lottery tasks continue to use multi-target approach as appropriate for their format
- **User-Created Group Taskings (August 15, 2025)**: Updated system to allow users to create both individual (1 credit) and group taskings (free) through the trial wizard. Group creators earn 1 credit when others join their tasks. Added max participant limits, enhanced navigation with separate "Create Task" and "Join Group" buttons, and integrated reward system for creators to incentivize collaboration
- **Scientific Case Study Integration (August 15, 2025)**: Added prominent University of Colorado Boulder research case study to homepage showcasing 100% stock market prediction success rate, featuring peer-reviewed study with real financial profits, historical ARV achievements ($250K+ profits), and enhanced visual design with gradient backgrounds and interactive elements to strengthen credibility and earning potential messaging
- **Enhanced Performance Dashboard (August 15, 2025)**: Completely redesigned dashboard with modern card-based layout, improved visual hierarchy using neutral color scheme, hover animations, better typography and iconography, responsive grid systems, and comprehensive CSS styling for professional appearance and optimal user experience
- **Auto-Generated Reddit-Style Usernames (August 14, 2025)**: Implemented automatic username generation for new user registrations using Reddit-style format like "ProViewer_2023" or "QuantumSeer_42", removing manual name input requirement and ensuring consistent community branding across the platform
- **Homepage Content Management System (August 14, 2025)**: Added dynamic homepage editor with HomepageContent database model, allowing admins to edit hero section, benefits, earning potential messaging, and leaderboard content through /admin/homepage interface with real-time updates
- **Logo Simplification (August 14, 2025)**: Removed eye icon from ARVLab logo in navigation bar for cleaner branding appearance
- **Target Upload Enhancement (August 14, 2025)**: Added comprehensive image size guidelines to target upload interface, recommending 400×400px to 800×800px square images for optimal display across all devices and responsive breakpoints
- **Terminology Update (August 14, 2025)**: Updated all user-facing text from "trial"/"trials" to "task"/"tasks" throughout templates, JavaScript, and user interfaces to better reflect the individual prediction nature of the ARV research platform
- **Website Rebrand (August 14, 2025)**: Successfully renamed website from "Quorum" to "Onsenses", then further rebranded to "ARVLab" across all interfaces including FastAPI app title, page titles, navigation branding, feedback messages, and documentation
- **Deployment Readiness (August 14, 2025)**: Fixed deployment issues by adding proper health check endpoints, supporting HEAD requests for `/` and `/health` routes, disabled problematic CLIP imports to prevent startup failures, added error handling for AI suggestion functions, and created deployment-ready startup script for Replit hosting
- **Comprehensive User Feedback System**: Implemented beautiful, functional feedback collection across all workflows with floating feedback button, star ratings, issue tracking, and admin dashboard
- **Workflow-Specific Feedback Tracking**: Added automatic feedback triggers for trial creation, ARV sessions, trial management, and admin workflows with context-aware prompting
- **Admin Feedback Dashboard**: Created comprehensive dashboard at `/admin/feedback` showing statistics, recent feedback, ratings analysis, and detailed feedback viewing with filtering capabilities
- **Enhanced Stock Trial Creation**: Added customizable outcome name fields to stock/binary trial wizard (e.g., "Stock Up/Stock Down" instead of generic "A/B") matching sports trial functionality
- **Enhanced AI Judging System**: Implemented three-tier analysis using pre-computed CLIP embeddings (primary), GPT-4o Vision API (fallback), and metadata analysis (final fallback) for maximum efficiency and accuracy
- **Visual Preprocessing Pipeline**: Added admin tool to pre-compute CLIP embeddings for all target images, enabling instant similarity matching during AI judging
- **RV Displacement Prevention**: Modified settled trial display to only show winning target image without tags, hiding losing outcomes completely to maintain ARV integrity for future sessions
- **Complete Tag Removal**: Eliminated all target tag displays from frontend interface to maintain ARV scientific integrity and prevent bias contamination
- **Enhanced Lottery Trial System**: Implemented comprehensive ball-specific descriptor functionality with dedicated creation and viewer wizards for systematic sensory impression collection across individual lottery balls
- **ARV Viewer Wizard**: Created specialized interface for remote viewers to systematically record impressions for each ball across all six sensory categories (Colors, Tactile, Energy, Smell, Sound, Visual) with progress tracking and auto-save functionality
- **Neutral Color Theming**: Implemented comprehensive neutral color scheme across entire application to prevent influencing remote viewing sessions - replaced all bright colors (primary blues, success greens, warning oranges, info blues, danger reds) with neutral grays and secondary colors throughout all templates, buttons, cards, badges, and status indicators
- **Modern ARV Wizard Design**: Completely redesigned ARV session interface to match trial detail page with modern card-based layout, rounded corners, subtle shadows, consistent neutral typography, and responsive ball navigation - removed individual "View Ball X" buttons and streamlined interface with beautiful visual consistency

### UI/UX Decisions
- **Brand Update**: Service name changed from "ConSenses ARV" to "Onsenses", then to "ARVLab" across all interfaces and documentation.
- **Terminology**: User-facing terminology updated from "Prediction Trials" to "Prediction Tasks".
- **Responsive Layout**: Implemented CSS Grid with adaptive breakpoints (mobile, small mobile, large screens) for optimal image display and flexible outcome cards, ensuring cross-device compatibility.
- **Visuals**: Photo-realistic target collection (151 distinct images + 100 stock photos) optimized for ARV distinctiveness.
- **User Experience Tiers**: Three-tier system (Trial Creator, Website Admin, Regular User) with role-based content protection (square placeholders, warnings) and participation integrity controls to maintain ARV scientific validity.

### Technical Implementations
- **Backend**: FastAPI with Python 3.x for async web API, Jinja2 for server-side rendering, WebSocket for real-time features.
- **Frontend**: Bootstrap 5 for responsive design, Feather Icons, vanilla ES6+ JavaScript for client-side interactivity.
- **Database**: SQLite for lightweight persistence, designed for easy migration to PostgreSQL.
- **Security**: Cookie-based sessions with `itsdangerous` signing, bcrypt password hashing, SHA-256 hashing for commit-reveal protocol, and FastAPI input validation.
- **Trial Lifecycle**: Supports draft, open, live, and settled states for prediction events.
- **Prediction Domains**: Binary (A/B) for stocks/general, Ternary (Win/Lose/Draw) for sports (with custom naming), and Lottery (2-20 balls) with dynamic outcome generation.
- **Deployment**: Configured for Replit Deployments with health check endpoints at `/health` and `/`, proper HEAD request support, graceful handling of optional dependencies (CLIP, AI features), and startup script for production deployment.
- **AI-Powered Features**:
    - **Automatic Target Selection**: Algorithm prioritizes maximal visual distinctiveness based on sensory contrasts (animate/inanimate, natural/artificial, etc.) and categorizes target tags into 12 visual categories.
    - **Trial Suggestions**: AI (via GPT-4o) analyzes trial configuration, timing, and viability to suggest improvements and optimize target selection based on ARV principles.
- **Collaborative Descriptors**: Structured descriptor system with 6 sensory categories (Colours, Tactile, Energy, Smell, Sound, Visual) with specific form fields, filtering, and batch submission.
- **Trial Management**:
    - **Trial Creation Wizard**: 5-step guided process with domain-specific forms, real-time validation, and smart target assignment.
    - **Lottery Creation Wizard**: Specialized 5-step wizard for enhanced lottery trials with ball configuration, timing setup, and automatic target assignment.
    - **ARV Viewer Wizard**: Dedicated interface at `/trials/{id}/arv-wizard` for systematic ball-by-ball sensory impression recording with progress tracking.
    - **Trial Edit Functionality**: Draft-only editing at `/trials/{id}/edit` with comprehensive forms, restricted to admin users.

### System Design Choices
- **Data Models**: Users (role-based), Trials, Targets (with optional CLIP embeddings), Outcomes, Predictions, Descriptors, Judgments, UserFeedback (workflow improvement tracking).
- **ARV Focus**: Design decisions prioritize scientific integrity for ARV experiments, including RV displacement prevention (showing only winning targets post-settlement) and protection against premature content viewing.

## External Dependencies

### Core Web Framework
- **FastAPI**
- **Uvicorn**
- **Jinja2**
- **python-multipart**

### Database & Storage
- **SQLModel** (ORM)
- **psycopg2-binary** (PostgreSQL adapter, for future migration)

### Authentication & Security
- **passlib** (bcrypt)
- **itsdangerous**

### AI & Machine Learning
- **OpenAI** (GPT, embeddings)
- **scikit-learn** (similarity calculations)
- **torch**
- **open_clip/clip** (image-text embeddings)
- **PIL (Pillow)** (image processing)

### Development & Configuration
- **python-dotenv**
- **Bootstrap 5** (CDN)
- **Feather Icons** (CDN)