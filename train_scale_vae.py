i=1
f=1
f_hat=0
while abs(f-f_hat)<0.001:
    mean, logvar = model.encoder_forward(sentences)