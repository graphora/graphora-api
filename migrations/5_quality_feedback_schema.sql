-- Quality Feedback Schema
-- This stores user feedback from quality validation reviews

CREATE TABLE IF NOT EXISTS quality_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    transform_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('quality_rejection', 'quality_approval', 'general_feedback')),
    feedback_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    source TEXT DEFAULT 'quality_dashboard',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_quality_feedback_user_id ON quality_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_quality_feedback_transform_id ON quality_feedback(transform_id);
CREATE INDEX IF NOT EXISTS idx_quality_feedback_type ON quality_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_quality_feedback_created_at ON quality_feedback(created_at);

-- Add comment to the table
COMMENT ON TABLE quality_feedback IS 'Stores user feedback from quality validation reviews';
COMMENT ON COLUMN quality_feedback.feedback_type IS 'Type of feedback: quality_rejection, quality_approval, or general_feedback';
COMMENT ON COLUMN quality_feedback.metadata IS 'Additional structured data about the feedback';
COMMENT ON COLUMN quality_feedback.source IS 'Source of the feedback (quality_dashboard, api, etc.)';