clear;
rng(13);
I=double(imread("barbara256.png"));
[H,W]=size(I);
E=I;
count = zeros([H,W]);
phi=randn([32,64],"double");
img2=zeros([H,W],'double');
B=phi;
alpha=eigs(B'*B,1);
qwert=1;
for i = 1:H-7
   for j = 1:W-7
       count(i:i+7,j:j+7) = count(i:i+7,j:j+7) + 1; 
       y = reshape(E(i:i+7,j:j+7), [],1);
       y=phi*y;
       d = double(ISTA_Harr(y,phi,alpha));
       img2(i:i+7,j:j+7) = img2(i:i+7,j:j+7) + d;
       disp(qwert);
       qwert=qwert+1;
   end
end
img2=img2./count;
figure;
imshow(mat2gray(img2));
RMSE = norm(E-img2)/norm(E);
disp(RMSE);
saveas(gcf,'./Images/Q2-C.png');

function img = ISTA_Harr(y,phi,alpha)
    [~,w] = size(phi);
    lambda=1;
    theta = eye([w,1],'double');
    new_theta=zeros([w,1],'double');
    time_taken =1;
    while (time_taken < 10 )
        theta=new_theta;
        [cA1,cH1,cV1,cD1] = func(theta);
        tempo=reshape(idwt2(cA1,cH1,cV1,cD1,'db1'),[64,1]);
        tempo1=y-phi*tempo;
        tempo2=phi'*tempo1;
        tempo2=reshape(tempo2,[8,8]);
        [cA1,cH1,cV1,cD1] = dwt2(tempo2,'db1');
        tempo3=rv2i(cA1,cH1,cV1,cD1);
        inter=theta+tempo3/alpha;
        new_theta=soft(inter,lambda/(2*alpha));
        time_taken=time_taken+1;
    end
    
    [cA1,cH1,cV1,cD1] = func(new_theta);
    img = idwt2(cA1,cH1,cV1,cD1,'db1');
end
function [cA1,cH1,cV1,cD1] = func(new_theta)
    cA1=reshape(new_theta(1:16,:),[4,4]);
    cH1=reshape(new_theta(17:32,:),[4,4]);
    cV1=reshape(new_theta(33:48,:),[4,4]);
    cD1=reshape(new_theta(49:64,:),[4,4]);
end
function rev_to_img = rv2i(l,m,n,o)
    l=reshape(l,[16,1]);
    m=reshape(m,[16,1]);
    n=reshape(n,[16,1]);
    o=reshape(o,[16,1]);
    rev_to_img=[l;m;n;o];
end
function out =soft(a,cons)
    [b,~]=size(a);
    out=zeros([b,1],'double');
    indices=(a>=cons);
    out(indices)=a(indices)-cons;
    indices=(a<=-cons);
    out(indices)=a(indices)+cons;
end