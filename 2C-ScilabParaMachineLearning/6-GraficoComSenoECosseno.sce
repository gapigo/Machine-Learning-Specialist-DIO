clf
x = [0:0.1:4*%pi];
y1 = sin(x);
y2 = cos(x);
plot(x, y1, '-*b');
plot(x, y2, '-*dr');
xtitle('Funções seno e cosseno');
xlabel('Eixo X');
ylabel('Eixo Y');
legend('Seno', 'Cosseno');
