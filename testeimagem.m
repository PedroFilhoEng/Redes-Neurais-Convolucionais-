
    I = imread('C:\Users\PF\Pictures\pneus10.jpg');
    I = imresize(I,[227,227]);
    label = classify(rede,I);
    imshow(I)
    title(char(label))
