**documentation/model_card.md**:
```markdown
## DistilBERT AG News Classification Model

### Model Details
- **Architecture**: DistilBERT-base-uncased
- **Task**: Text Classification
- **Classes**: 4 (World, Sports, Business, Sci/Tech)
- **Training Data**: AG News Dataset (120k samples)
- **Evaluation Data**: AG News Test Set (7.6k samples)

### Performance
| Metric     | Score |
|------------|-------|
| Accuracy   | 94.2% |
| Weighted F1| 94.1% |

### Recommended Use
- News article classification
- Text categorization experiments
- Transfer learning base for similar tasks

### Limitations
- English language only
- News domain specific
- Max sequence length: 128 tokens
