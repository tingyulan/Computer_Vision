%str='C:\Users\Administrator\Documents\MATLAB\task1and2_hybrid_pyramid';
str1='D:\WIN10\Documents\MATLAB\task1and2_hybrid_pyramid\makeup_after.jpg';
str2='D:\WIN10\Documents\MATLAB\task1and2_hybrid_pyramid\makeup_before.jpg';
ideal(str1,str2)
figure
gaussian(str1,str2)
function ideal(str1,str2)
    img=imread(str1);
    img = rgb2gray(img);

    [M, N] = size(img);
    P = 2 * M;
    Q = 2 * N;

    img = double(img);

    img_fp = zeros(P, Q);
    img_fp(1:M, 1:N) = img(1:M, 1:N);



    for x = 1:P
        for y = 1:Q
            img_fp(x, y) = img_fp(x, y) .* (-1)^(x+y);
        end
    end

    img_Fp = fft2(img_fp);


    D0 = 30;
    H = ones(P, Q);
    for x = 1:P
        for y = 1:Q
            D = sqrt((x-M)^2 + (y-N)^2);
            if D > D0
                H(x, y) = 0;
            else
                H(x, y) = 1;
            end

        end
    end

    img_G = img_Fp .* H;

    img_g = ifft2(img_G);
    img_g = real(img_g);

    for x = 1:P
        for y = 1:Q
            img_g(x, y) = img_g(x, y) .* (-1)^(x+y);
        end
    end

    img_o1 = img_g(1:M, 1:N);

    img=imread(str2);
    img = rgb2gray(img);

    [M, N] = size(img);

    P = 2 * M;
    Q = 2 * N;

    img = double(img);

    img_fp = zeros(P, Q);
    img_fp(1:M, 1:N) = img(1:M, 1:N);



    for x = 1:P
        for y = 1:Q
            img_fp(x, y) = img_fp(x, y) .* (-1)^(x+y);
        end
    end

    img_Fp = fft2(img_fp);


    
    H = ones(P, Q);
    for x = 1:P
        for y = 1:Q
            D = sqrt((x-M)^2 + (y-N)^2);
            if D > D0
                H(x, y) = 1;
            else
                H(x, y) = 0;
            end

        end
    end

    img_G = img_Fp .* H;

    img_g = ifft2(img_G);
    img_g = real(img_g);

    for x = 1:P
        for y = 1:Q
            img_g(x, y) = img_g(x, y) .* (-1)^(x+y);
        end
    end

    img_o = img_g(1:M, 1:N);
    [h1,w1]=size(img_o);
    [h2,w2]=size(img_o1);
    if(h1>h2)
        img_o(h2+1:h1,:)=[];
    elseif(h2>h1)
        img_o1(h1+1:h2,:)=[];
    end
    [h1,w1]=size(img_o);
    [h2,w2]=size(img_o1);
    if(w1>w2)
        img_o(:,w2+1:w1)=[];
    elseif(w2>w1)
        img_o1(:,w1+1:w2)=[];
    end
    imshow(img_o+img_o1,[]);
end
function gaussian(str1,str2)
    img=imread(str1);
    img=rgb2gray(img);
    [M, N] = size(img);
    P = 2 * M;
    Q = 2 * N;
    img = double(img);
    img_fp = zeros(P, Q);
    img_fp(1:M, 1:N) = img(1:M, 1:N);

    for x = 1:P
        for y = 1:Q
            img_fp(x, y) = img_fp(x, y) .* (-1)^(x+y);
        end
    end

    img_Fp = fft2(img_fp);


    D0 = 30;
    H = ones(P, Q);
    for x = 1:P
        for y = 1:Q
            d = sqrt((x-M)^2 + (y-N)^2);
            H(x,y)=exp(-(d^2)/(2*D0*D0));
        end
    end

    img_G = img_Fp .* H;
    img_g = ifft2(img_G);
    img_g = real(img_g);

    for x = 1:P
        for y = 1:Q
            img_g(x, y) = img_g(x, y) .* (-1)^(x+y);
        end
    end

    img_o1 = img_g(1:M, 1:N);

    img=imread(str2);
    img=rgb2gray(img);

    [M, N] = size(img);

    P = 2 * M;
    Q = 2 * N;

    img = double(img);

    img_fp = zeros(P, Q);
    img_fp(1:M, 1:N) = img(1:M, 1:N);



    for x = 1:P
        for y = 1:Q
            img_fp(x, y) = img_fp(x, y) .* (-1)^(x+y);
        end
    end

    img_Fp = fft2(img_fp);


    
    H = ones(P, Q);
    for x = 1:P
        for y = 1:Q
            d = sqrt((x-M)^2 + (y-N)^2);
            H(x,y)=1-exp(-(d^2)/(2*D0*D0));

        end
    end

    img_G = img_Fp .* H;

    img_g = ifft2(img_G);
    img_g = real(img_g);

    for x = 1:P
        for y = 1:Q
            img_g(x, y) = img_g(x, y) .* (-1)^(x+y);
        end
    end

    img_o = img_g(1:M, 1:N);
    [h1,w1]=size(img_o);
    [h2,w2]=size(img_o1);
    if(h1>h2)
        img_o(h2+1:h1,:)=[];
    elseif(h2>h1)
        img_o1(h1+1:h2,:)=[];
    end
    [h1,w1]=size(img_o);
    [h2,w2]=size(img_o1);
    if(w1>w2)
        img_o(:,w2+1:w1)=[];
    elseif(w2>w1)
        img_o1(:,w1+1:w2)=[];
    end
    imshow(img_o+img_o1,[]);
end


