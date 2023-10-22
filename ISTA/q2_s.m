clear;
rng(124);
I=double(imread("barbara256.png"));
[H,W]=size(I);
%% A part
% Add noise
E = I + randn([H,W])*4;
figure;
imshow(mat2gray(E));

img = zeros([H,W],'double');
count = zeros([H,W]);
psi = kron(dctmtx(8)',dctmtx(8)');
A=eye(64,64);
B=A*psi;
alpha=2; % 2* max eigenvalue of Identity matrix

% Call ISTA for patches
for i = 1:H-7
   for j = 1:W-7
       count(i:i+7,j:j+7) = count(i:i+7,j:j+7) + 1; 
       y = reshape(E(i:i+7,j:j+7), [],1);
       d = double(ISTA(y,B,psi,alpha, 1, 1));
       d=reshape(d,[8,8]);
       img(i:i+7, j: j+7) = img(i:i+7,j:j+7) + d;
   end
end
img = img./count;

% Show reconstructed
figure;
imshow(mat2gray(img));
RMSE = norm(E-img)/norm(E);
disp(RMSE);
saveas(gcf,'./Images/Q2-A.png');

%% B part
phi=randn([32,64],"double");
img=zeros([H,W],'double');
E=I;
B=phi*psi;
alpha = 3*eigs(B'*B, 1);
count1 = 0;
for i = 1:H - 7
   for j = 1:W - 7
       y = reshape(E(i:i+7,j:j+7), [],1);
       y=phi*y;
       d = double(ISTA(y, B, psi, alpha, 0.05, 1));
       d=reshape(d,[8,8]);
       img(i:i+7, j: j+7) = img(i:i+7,j:j+7) + d;
   end
end
img= img./count;
figure;
imshow(mat2gray(img));
RMSE = norm(E-img)/norm(E);
disp(RMSE);
saveas(gcf,'./Images/Q2-B.png');

%% D part
n = 100;
k = 10;
vec = zeros(n,1);
idx = randperm(n);
vec(idx(1:k)) = randn(k,1)';
figure;
stem(vec);
saveas(gcf,'./Images/Q2-D-OriginalSparseImage.png');
h = [1,2,3,4,3,2,1]/16;
magnitude = 0.05* norm(vec);
convol = conv(h,vec);
y = convol + magnitude*randn([size(convol,1),1]);
figure;
stem(y);
saveas(gcf,'./Images/Q2-D-NoisySparseImage.png');
A = convmtx(h, n)';
alpha = eigs(A'*A, 1);
reconstructed = ISTA(y, A, eye(size(n,1)), alpha, 0.0000001, 0.1);
figure;
stem(reconstructed);
saveas(gcf,'./Images/Q2-D-ReconstructedImage.png');
%% Functions
function img = ISTA(y, B, psi, alpha, epsilon, lambda)
    [~,w] = size(B);
    c = lambda/(2*alpha);
    new_theta=zeros([w,1],'double');
    theta = randn([w,1], 'double');
    while (norm(new_theta-theta)> epsilon)
        theta = new_theta;
        new_theta = soft(theta+((B'*(y-B*theta))/alpha),c);
    end
    img = psi*new_theta;
end
function out = soft(a,cons)
    [b,~]=size(a);
    out=zeros([b,1],'double');
    i1 = a >= cons; i2 = a <= -cons;
    out(i1) = a(i1) - cons;
    out(i2) = a(i2) + cons;
end