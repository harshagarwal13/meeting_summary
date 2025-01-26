import re
from transformers import pipeline, AutoTokenizer
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeetingSummarizer:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.action_item_model = pipeline(
            "text2text-generation",
            model="t5-small",
            tokenizer=self.tokenizer
        )

        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def process_transcript(self, raw_text: str) -> str:
        try:
            # Remove timestamps and special characters
            cleaned = re.sub(r'\d{2}:\d{2}:\d{2}', '', raw_text)
            cleaned = re.sub(r'\[.*?\]', '', cleaned)

            # Identify speakers
            cleaned = re.sub(r'(\w+):', r'\n\1:', cleaned)
            return cleaned.strip()
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            raise

    def generate_summary(self, text: str) -> str:
        try:
            max_length = 1024  # Model's max input length
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)

            return " ".join(summaries)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def extract_action_items(self, text: str) -> List[Dict]:
        try:
            prompt = f"""
            Extract action items from this meeting transcript. 
            Follow format: [Person] needs to [Task] by [Deadline].
            Transcript: {text[:2000]}  # Truncate for model limits
            """

            result = self.action_item_model(
                prompt,
                max_length=500,
                num_beams=4,
                early_stopping=True
            )[0]['generated_text']

            return self._parse_action_items(result)
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []

    def _parse_action_items(self, text: str) -> List[Dict]:
        items = []
        for line in text.split('\n'):
            if 'needs to' in line:
                parts = line.split(' needs to ')
                person = parts[0].strip('[]')
                task_deadline = parts[1].split(' by ')
                task = task_deadline[0]
                deadline = task_deadline[1].strip('.') if len(task_deadline) > 1 else None

                items.append({
                    "person": person,
                    "task": task,
                    "deadline": deadline
                })
        return items

    def detect_disagreements(self, text: str) -> Dict:
        try:
            disagreement_threshold = 0.7  # Confidence threshold
            sentences = re.split(r'(?<=[.!?]) +', text)

            conflicts = []
            for sent in sentences:
                if len(sent.split()) > 3:  # Skip short phrases
                    result = self.sentiment_analyzer(sent)[0]
                    if result['label'] == 'NEGATIVE' and result['score'] > disagreement_threshold:
                        conflicts.append(sent)

            return {
                "conflict_points": conflicts,
                "count": len(conflicts)
            }
        except Exception as e:
            logger.error(f"Error detecting disagreements: {e}")
            return {"conflict_points": [], "count": 0}


def main():
    summarizer = MeetingSummarizer()

    sample_transcript = """
    09:00:00 John: Let's discuss Q4 goals. I think we should aim for 20% growth.
    09:00:05 Sarah: That's too ambitious! We barely achieved 10% last quarter.
    09:00:10 Mike: I agree with John. With the new marketing budget...
    09:00:15 Sarah: The marketing budget is already overstretched.
    09:00:20 John: Sarah, can you prepare a budget report by Friday?
    """

    try:
        cleaned = summarizer.process_transcript(sample_transcript)
        summary = summarizer.generate_summary(cleaned)
        action_items = summarizer.extract_action_items(cleaned)
        conflicts = summarizer.detect_disagreements(cleaned)

        print("## Meeting Summary ##")
        print(summary)

        print("\n## Action Items ##")
        for item in action_items:
            print(f"- {item['person']}: {item['task']} ({item['deadline'] or 'no deadline'})")

        print("\n## Potential Disagreements ##")
        print(f"Found {conflicts['count']} conflict points:")
        for point in conflicts['conflict_points']:
            print(f"- {point}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()