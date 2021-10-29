% function calc_EGM_1D(matname,movieflag,n)
% calculates external electrical fields (phie) like the ones recorded from
% electrograms in isotropic conditions
% matname: name of file with V field and scalar D field
% n: number of electrodes in each row
% movieflag: 0 no display; 1 display traces (n<=4); 2 display image
% Marta, 11/06/2021

load(matname);

% sizes
tini=2;
sz=size(Vsav); %2D data
X=sz(1);
tmax=sz(2);

if numel(D)==1 % D is a scalar
    D=D*ones(X,1);
elseif size(D,1)==sz(1)+2
    D=D(2:end-1,1); 
end

% identify heterogeneous regions in D
if any(diff(D(:)))
    binD=logical(D>mean(D(:)));
    [B,L,N,A] = bwboundaries(~binD,'holes');
    B=B{1};
else
    B(1:2,1:2)=NaN;
end

h=1;
Dx=gradient(D,h);
    
% set up electrode positions
% n=10;

xel=linspace(10,X-10,n)+0.5;
dispmat=zeros(n,1);

yval=0.05; % electrogram scale

if n<=16
    col={'b','k','r','m','c','g','y','b','k','r','m','c','g','y','b','k'};
    col=col(1:n);
elseif movieflag==1
    movieflag=2;
end
   

if movieflag>0
    moviename=[matname '_EGM'];
    writerObj = VideoWriter(moviename);
    writerObj.FrameRate = 3;
    open(writerObj);
end

tt=0;
figure
for t=tini:tmax
    tt=tt+1;
    V=squeeze(Vsav(:,t));
    %du=D*del2(V);
    
    gx=gradient(V,h);
    
    du=4*D.*del2(V,h)+Dx.*gx; 

    
    if movieflag==1
        subplot(n*2,1,1:n)
    elseif movieflag==2
        subplot(2,1,1)
    end
    imagesc(repmat(V',[round(X/20) 1]),[-0 1])
    hold all
    plot(B(:,2),B(:,1),'k.')
    colorbar
    text(5,5,['t:' num2str(tfac*t)],'color','k','FontSize',16)     
    axis image
    hold off
    
    % calculate phie
    for k=1:n
        matx=(2:X-1)-xel(k);
        phie(k,tt)=-sum(du(2:X-1)'./matx); 
    end
    
    if movieflag==1
        for j=1:n
            subplot(n*2,1,n+j)
            plot(tini:t,squeeze(phie(j,1:tt)),'-','LineWidth',2,'Color',col{j})
            hold all
            ylabel([num2str(j)],'Color',col{j})
            set(gca,'YTick',[])
            xlim([tini t+1])
    %                     ylim([-1 1])
            set(gca,'YTick',-yval*5:yval*5:yval*5)

            if j~=n
                set(gca,'XTickLabel',[])
            end
            grid on
        end
        xlabel('Time')
%         waitforbuttonpress;

        frame = getframe(gcf);
        writeVideo(writerObj,frame);
    elseif movieflag==2
        subplot(2,1,2)
        ph=phie(:,tt);
        matdisp=ph;
        imagesc(matdisp',[-yval yval])
        hold all
        plot(B(:,2)/10,B(:,1)/10,'k.')
        hold off
%         axis off
        axis image
        colorbar
        
        frame = getframe(gcf);
        writeVideo(writerObj,frame);
    end
end

 if movieflag
    close(writerObj);   
 end

% figure
% for j=1:n2
%     subplot(n2,1,j)
%     plot(tini:tmax,squeeze(phie(j,:)),'-','LineWidth',2,'Color',col{j})
%     ylabel([num2str(j)],'Color',col{j})
%     set(gca,'YTick',[])
%     xlim([tini tmax])
%     ylim([-1 1])
%     set(gca,'YTick',-1:1:1)
% 
%     if j~=n2m
%         set(gca,'XTickLabel',[])
%     end
%     grid on
%     if j==1
%         title(matname)
%     end
% end
% xlabel('Time')
% saveas(gcf,[matname '_EGM.png']);
% saveas(gcf,[matname '_EGM.fig']);
save([matname(1:end-4) '_EGM.mat'])

% end