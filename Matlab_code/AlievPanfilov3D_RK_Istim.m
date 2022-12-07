% function [Vsav,Wsav]=AlievPanfilov2D_RK_Istim(BCL,ncyc,extra,ncells,iscyclic,flagmovie)
% (Ventricular) Aliev-Panfilov model in single-cell with the parameters
% from Goektepe et al, 2010
% Marta, 23/04/2021

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

close all
clear all
BCL=100;
ncyc=2;
extra=0;
ncells=100;
thick=10;
iscyclic=0;
flagmovie=1;

% one of the biggest determinants of the propagation speed
% (D should lead to realistic conduction velocities, i.e.
% between 0.6 and 0.9 m/s)
X = ncells + 2; % to allow boundary conditions implementation
Y = ncells + 2;
Z = thick + 2;
stimgeo=false(X,Y,Z);
stimgeo(1:5,1:5,1:5)=true; % indices of cells where external stimulus is felt

% Model parameters
% time step below 10x larger than for forward Euler
dt=0.005; % AU, time step for finite differences solver
gathert=round(1/dt); % number of iterations at which V is outputted
% for plotting, set to correspond to 1 ms, regardless of dt
tend=BCL*ncyc+extra; % ms, duration of simulation
stimdur=1; % UA, duration of stimulus
Ia=0.1*stimgeo; % AU, value for Istim when cell is stimulated

V(1:X,1:Y,1:Z)=0; % initial V
W(1:X,1:Y,1:Z)=0.01; % initial W

Vsav=zeros(ncells,ncells,thick,ceil(tend/gathert)); % array where V will be saved during simulation
Wsav=zeros(ncells,ncells,thick,ceil(tend/gathert)); % array where W will be saved during simulation

ind=0; %iterations counter
kk=0; %counter for number of stimuli applied

y=zeros(2,size(V,1),size(V,2),size(V,3));

% for loop for explicit RK4 finite differences simulation
for t=dt:dt:tend % for every timestep
    ind=ind+1; % count interations
        % stimulate at every BCL time interval for ncyc times
        if t>=BCL*kk&&kk<ncyc
            Istim=Ia; % stimulating current
        end
        % stop stimulating after stimdur
        if t>=BCL*kk+stimdur*2
            kk=kk+1;
            Istim=zeros(X,Y); % stimulating current
        end
        
        y(1,:,:,:)=V;
        y(2,:,:,:)=W;
        k1=AlPan(y,Istim);
        k2=AlPan(y+dt/2.*k1,Istim);
        k3=AlPan(y+dt/2.*k2,Istim);
        k4=AlPan(y+dt.*k3,Istim);
        y=y+dt/6.*(k1+2*k2+2*k3+k4);
        V=squeeze(y(1,:,:,:));
        W=squeeze(y(2,:,:,:));
                      
        % rectangular boundary conditions: no flux of V
        if  ~iscyclic % 1D cable
            V(1,:,:)=V(2,:,:);
            V(end,:,:)=V(end-1,:,:);
            V(:,1,:)=V(:,2,:);
            V(:,end,:)=V(:,end-1,:);
            V(:,:,1)=V(:,:,2);
            V(:,:,end)=V(:,:,end-1);
        else % periodic boundary conditions in x, y or both
            % set up later - need to amend derivatives calculation too
        end
        
        % At every gathert iterations, save V value for plotting
        if mod(ind,gathert)==0
            % save values
            Vsav(:,:,:,round(ind/gathert))=permute(V(2:end-1,2:end-1,2:end-1),[2 1 3]);
            Wsav(:,:,:,round(ind/gathert))=permute(W(2:end-1,2:end-1,2:end-1),[2 1 3]);
            % show (thicker) cable
            if flagmovie
                subplot(2,1,1)
                imagesc(V(2:end-1,2:end-1,5)',[0 1])
                axis image
                set(gca,'FontSize',14)
                xlabel('x (voxels)')
                ylabel('y (voxels)')
                set(gca,'FontSize',14)
                title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                colorbar
                
                subplot(2,1,2)
                imagesc(W(2:end-1,2:end-1,5)',[0 1])
                axis image
                set(gca,'FontSize',14)
                xlabel('x (voxels)')
                ylabel('y (voxels)')
                set(gca,'FontSize',14)
                title(['V (AU) - Time: ' num2str(t,'%.0f') ' ms'])
                colorbar
                set(gca,'FontSize',14)
                title('W (AU)')
                colorbar
                pause(0.01)
            end
        end
end
close all

function dydt = AlPan(y,Istim)
    a = 0.01;
    k = 8.0;
    mu1 = 0.2;
    mu2 = 0.3;
    epsi = 0.002;
    b  = 0.15;
    h = 0.1; % mm cell length
    D = 0.05; % mm^2/UA, diffusion coefficient (for monodomain equation)
    
    V=squeeze(y(1,:,:,:));
    W=squeeze(y(2,:,:,:));
    dV=4*D.*del2(V,h);
    dWdt=(epsi + mu1.*W./(mu2+V)).*(-W-k.*V.*(V-b-1));
    dVdt=(-k.*V.*(V-a).*(V-1)-W.*V)+dV+Istim;
    dydt(1,:,:,:)=dVdt;
    dydt(2,:,:,:)=dWdt;
end
% end
