### BGCL_KDE



##### Last.FM

```python
python Main.py --data music --e_loss 0.1 --temp 1.0 --ssl_reg 0.001 --mess_dropout_rate 0.2 --res_lambda 1 --epsilon 0.5 --epoch 100 --rebuild_k 4 --steps 10
```

##### AmazonBook

```python
python Main.py --data book --e_loss 0.1 --temp 0.5 --ssl_reg 1.0 --mess_dropout_rate 0.2 --res_lambda 1 --epsilon 0.5 --epoch 25 --rebuild_k 2 --steps 5 --similarity 80_10_1
```

##### Movielens

```python
python Main.py --data movie --e_loss 0.1 --temp 0.5 --ssl_reg 0.01 --mess_dropout_rate 0.2 --res_lambda 1 --epoch 50 --epsilon 0.4 --steps 5 --similarity 80_8_2
```

##### luo

```python
python Main.py --data luo --e_loss 0.1 --temp 1.0 --ssl_reg 0.01 --mess_dropout_rate 0.2 --res_lambda 1 --topk 50 --epsilon 0.4 --rebuild_k 1 --steps 5 --similarity 80_10_2 --drug_pattern 1
```

