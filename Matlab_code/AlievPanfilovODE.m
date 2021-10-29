function AlievPanfilovODE
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% from Goektepe et al, 2010
% using ode45
% Marta, 18/03/2020
% see AlievPanfilov0D_forClara.m for equivalent using forward Euler

% BCL in AU: basic cycle length: time between repeated stimuli (e.g. 30)
% ncyc: number of cycles, number of times the cell is stimulated (e.g. 10)
% extra in AU: time after BCL*ncyc during which the simulation runs (e.g.
% 0)
% ncells is number of cells in 1D cable (e.g. 200)
% iscyclic, = 0 for a cable, = 1 for a ring (connecting the ends of the
% cable - the boundary conditions are not set for the ring yet!)
% flagmovie, = 0 to show a movie of the potential propagating, = 0
% otherwise

% Aliev-Panfilov model parameters 
% V is the electrical potential difference across the cell membrane in 
% arbitrary units (AU)
% t is the time in AU - to scale do tms = t *12.9


[t,y]=ode45(@AlPan,[0; 100],[1;0]);

function dydt = AlPan(t,y)
    a = 0.01;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    b  = 0.15;
    V=y(1);
    W=y(2);
    dWdt=(epsi + mu1*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dV=0;
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV;
    dydt=[dVdt; dWdt];
end

hold all
plot(t,y(:,1),':','LineWidth',2)
% plot(t,y(:,2),'r--s','LineWidth',2)
% legend('V (AU)','W (AU)','Location','Best')
xlabel('Time (AU)')
set(gca,'FontSize',14)
grid on
end
                      
        