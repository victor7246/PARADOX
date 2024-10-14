# PARADOX
Code for Persona-Aware Code-Mixed Text Generation

### How to run

	python src/train.py \
    --data_file ./data/twitter_data.csv \
    --model_save_path ./models/ \
    --max_text_len 40 \
    --use_fame \
    --use_persona \
    --use_alignment

### Citation

If you find this code useful, please consider giving a star and citation:
```bibtex
@article{sengupta2024paradox,
  title={Persona-aware Generative Model for Code-mixed Language},
  author={Sengupta, Ayan and Akhtar, Md. Shad and Chakraborty, Tanmoy},
  year={2024}
}
```