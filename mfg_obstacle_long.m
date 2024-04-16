clearvars 

%%% Settings 

graphics = 1;
save = 0;
mix = 0.1;
tol = 10e-3;

%%% Main parameters

xi = 0.4;
c_s = 0.2;
alpha = 0;
gam = 6;
nx = 80;
ny = 120;
sup_x = 2;
sup_y = 3;

%%%Fixed Parameters 

s = 0.6;
m_0 = 3.5; 
mu = 1;
R = 0.37;
g = -(2*mu*c_s^2)/m_0;
sigma = sqrt(2*c_s*xi);

%%% Spatial Discretisation

box_x = sup_x +1;
box_y = sup_y +1 ;
y_start = sup_y-1;
y_end = -y_start;
inf_x = -sup_x;
inf_y = -sup_y;
lx = sup_x - inf_x;
ly = sup_y - inf_y;
dx = lx/(nx - 1);
dy = ly/(ny - 1);
[X,Y] = meshgrid(linspace(inf_x,sup_x,nx),linspace(inf_y,sup_y,ny));


%%% Potential limit 

V_lim = (mu*sigma^4)/(2*max(dx,dy)^2);
V = V_lim*0.95;

%%% Temporal Discretisation  

t_0 = 0;
T = abs(y_end-y_start)/s;

nt = 100; 
dt = (T-t_0)/(nt-1);

%%% Initial Conditions 

u_final = zeros(nx,ny);
m_init = zeros(nx,ny);
for i=(1:nx)
    for j=(1:ny)
        x = inf_x + (i-1)*dx;
        y = inf_y + (j-1)*dy;
        d = sqrt(x^2+(y+3)^2);
        if abs(x) < box_x && abs(y) < box_y
            m_init(i,j) = m_0;
        end
    end 
end
u_total = zeros(nt,nx*ny);
m_total = zeros(nt,nx*ny);
u_total(1,:) = reshape(u_final,nx*ny,1);
for t = (1:nt)
    m_total(t,:) = reshape(m_init,nx*ny,1);
end
m_old = zeros(nt,nx*ny);

%%% Integration

err = 1;
cycle = 0;
t_start = tic;
while err > tol
lim = sup_y;
m_old = m_total;
u_old = u_total;
for i = (1:nt-1)
    [t,u] = ode45(@(t,u) hjb(t,u,m_total(nt-i+1,:)',dx,dy,nx,ny,gam,mu,g,sigma,V,inf_x,inf_y,y_start,s,R,alpha,box_x,box_y),[T-(i-1)*dt T-i*dt],u_total(i,:)');
    if graphics ==1 
        h = pcolor(X',-Y',reshape(u(end,:),nx,ny))';set(h, 'EdgeColor', 'none');xlabel('x');ylabel('y');
        timestamp = sprintf("Step number " + i);
        text(sup_x-0.5, sup_y-0.5, timestamp,"HorizontalAlignment","right");
        xlim([-sup_x,sup_x]);ylim([-sup_y,sup_y]);
        colorbar
        drawnow
    end
    u_total(i+1,:) = u(end,:);
end
for j=(1:nt-1)
    [t,m] = ode45(@(t,m) kfp(t,u_total(nt-j+1,:)',m,dx,dy,nx,ny,mu,sigma,alpha),[(j-1)*dt j*dt],m_total(j,:)');
    if graphics ==1 
%         h = pcolor(X',-Y',reshape(m(end,:),nx,ny));set(h, 'EdgeColor', 'none');xlabel('x');ylabel('y');
%         colorbar
        timestamp = sprintf("Step number " + j);
        text(sup_x-0.5, sup_y-0.5, timestamp,"HorizontalAlignment","right");
        xlim([-sup_x,sup_x]);ylim([-sup_y,sup_y]);
        ergo = reshape(m(end,:),nx,ny);
        plot(ergo(nx/2,:))
        drawnow
    end
    m_total(j+1,:) = m(end,:);
end
err = sqrt(sum(sum((m_total - m_old).^2))/sum(sum(m_old.^2)));
u_keep = u_total;
m_keep = m_total;
u_total = mix.*u_total + (1-mix).*u_old;
m_total = mix.*m_total + (1-mix).*m_old;
fprintf("Cycle number = %d \n", cycle);
fprintf("Error = %e \n", err);
cycle = cycle +1;
end
u_total = u_keep;
m_total = m_keep;
t_end = toc(t_start);
fprintf("Elapsed time = " + floor(t_end/3600) + "h " + floor(t_end/60) + "m " + int64(rem(t_end,60)) + "s");

if save == 1
    m_filename= sprintf("data/m_evo_xi=" +xi + "_cs=" + c_s+ "_alpha=" +alpha + "_gamma=" +gam+"_"+ nx + "x" + ny + "_"+ sup_x + "x" + sup_y + ".txt");
    u_filename= sprintf("data/u_evo_xi=" +xi + "_cs=" + c_s+ "_alpha=" +alpha + "_gamma=" +gam+"_"+ nx + "x" + ny + "_"+ sup_x + "x" + sup_y + ".txt");
    writematrix(m_total,m_filename)
    writematrix(u_total,u_filename)
end

m = max(max(m_total));

%%% Functions Definition

function exit_u = hjb(t,u,m,dx,dy,nx,ny,gam,mu,g,sigma,V,inf_x,inf_y,y_start,s,R,alpha,box_x,box_y)
temp_u = reshape(u,nx,ny);
temp_m = reshape(m,nx,ny);
exit_u = zeros(nx,ny);
for i=(1:nx)
    for j=(1:ny)
        c = C(t,inf_x,inf_y,dx,dy,y_start,s,V,R,i,j,box_x,box_y);
        i_plus = int16(mod(i,nx)+1);
        i_minus = int16(mod(i-2,nx)+1);
        j_plus = int16(mod(j,ny)+1);
        j_minus = int16(mod(j-2,ny)+1);
        lap_u = (temp_u(i_plus,j) + temp_u(i,j_plus) + temp_u(i_minus,j) + temp_u(i,j_minus) - 4*temp_u(i,j))/(dx*dy);
        grad_u_x = (temp_u(i_plus,j)-temp_u(i_minus,j))/(2*dx);
        grad_u_y = (temp_u(i,j_plus)-temp_u(i,j_minus))/(2*dy);
        exit_u(i,j) = -sigma^2*0.5*lap_u + (grad_u_x^2 + grad_u_y^2)/(2*mu*(1+alpha*temp_m(i,j)))...
        + temp_u(i,j)*gam + (g*temp_m(i,j) + c);
    end
end
exit_u = reshape(exit_u,nx*ny,1);
end

function exit_m = kfp(~,u,m,dx,dy,nx,ny,mu,sigma,alpha)
temp_u = reshape(u,nx,ny);
temp_m = reshape(m,nx,ny);
exit_m = zeros(nx,ny);
for i=(1:nx)
    for j=(1:ny)
        i_plus = int16(mod(i,nx)+1);
        i_minus = int16(mod(i-2,nx)+1);
        j_plus = int16(mod(j,ny)+1);
        j_minus = int16(mod(j-2,ny)+1);
        grad_u_x = (temp_u(i_plus,j)-temp_u(i_minus,j))/(2*dx);
        grad_u_y = (temp_u(i,j_plus)-temp_u(i,j_minus))/(2*dy);
        grad_m_x = (temp_m(i_plus,j)-temp_m(i_minus,j))/(2*dx);
        grad_m_y = (temp_m(i,j_plus)-temp_m(i,j_minus))/(2*dy);
        lap_m = (temp_m(i_plus,j) + temp_m(i,j_plus) + temp_m(i_minus,j) + temp_m(i,j_minus) - 4*temp_m(i,j))/(dx*dy);
        lap_u = (temp_u(i_plus,j) + temp_u(i,j_plus) + temp_u(i_minus,j) + temp_u(i,j_minus) - 4*temp_u(i,j))/(dx*dy);
        exit_m(i,j) = sigma^2*0.5*lap_m + (grad_m_x*grad_u_x + grad_m_y*grad_u_y)/(mu*(1+alpha*temp_m(i,j))^2)... 
        + (temp_m(i,j)*lap_u)/(mu*(1+alpha*temp_m(i,j)));
    end
end
exit_m = reshape(exit_m,nx*ny,1);
end 

function cylinder = C(t,inf_x,inf_y,dx,dy,y_start,s,V,R,i,j,box_x,box_y)
x = inf_x + (i-1)*dx;
y = inf_y + (j-1)*dy;
d = sqrt(x^2 + (y - y_start + s*t)^2);
if d < R
    cylinder = -V;
elseif abs(x) < box_x && abs(y) < box_y && d > R
    cylinder = 0;
else 
    cylinder = -V;
end
end

