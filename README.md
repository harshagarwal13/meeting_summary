# Meeting Summarizer Documentation

## Overview

The `MeetingSummarizer` is a Python-based tool that processes meeting transcripts to:
1. Generate summaries of meeting discussions.
2. Extract actionable items with deadlines.
3. Detect potential points of disagreement or conflict.

It uses state-of-the-art machine learning models and NLP pipelines to achieve these functionalities.

---

## Features

### 1. **Transcript Cleaning**
   - Removes timestamps, special characters, and unnecessary formatting from the transcript.
   - Identifies and separates speakers for better readability.

### 2. **Summarization**
   - Summarizes lengthy meeting transcripts into concise text.
   - Uses the `facebook/bart-large-cnn` model for high-quality summarization.
   - Handles large transcripts by splitting them into manageable chunks.

### 3. **Action Item Extraction**
   - Extracts action items in the format: `[Person] needs to [Task] by [Deadline]`.
   - Leverages the `t5-small` model to generate actionable insights.

### 4. **Conflict Detection**
   - Analyzes text sentiment to identify negative or conflicting statements.
   - Detects sentences with high negative sentiment (confidence > 0.7).

---

## Requirements

### Dependencies
- `transformers`
- `re`
- `logging`

### Install Required Libraries
```bash
pip install transformers
pip install regex
