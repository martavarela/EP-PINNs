function calc_EGM_2D(matname,movieflag,n)
% calculates external electrical fields (phie) like the ones recorded from
% electrograms in isotropic conditions
% matname: name of file with V field and scalar D field
% n: number of electrodes in each row
% movieflag: 0 no display; 1 display traces (n<=4); 2 display image
% Marta, 11/06/2021

load(matname);

% sizes
sz=size(Vsav);
tmax=sz(3); %sz(1);
tini=2;
X=sz(1);
Y=sz(2);

if numel(D)==1 % D is a scalar
    D=D*ones(X,Y); % make homogeneous scalar field
elseif size(D,1)==sz(1)+2&&size(D,2)==sz(2)+2
    D=D(2:end-1,2:end-1); 
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
[Dx,Dy]=gradient(D,h);

% set up electrode positions
% n=10;
n2=n^2;

el=linspace(10,X-10,n)+0.5;
[xel,yel]=meshgrid(el);
dispmat=zeros(n,n);

yval=0.1; % electrogram scale

if n<=4
    col={'b','k','r','m','c','g','y','b','k','r','m','c','g','y','b','k'};
    col=col(1:n2);
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
    V=squeeze(Vsav(:,:,t));
    %du=D*del2(V);
    
    [gx,gy]=gradient(V,h);
    
    du=4*D.*del2(V,h)+Dx.*gx+Dy.*gy; 

    
    if movieflag==1
        subplot(n2*2,1,1:n2)
    elseif movieflag==2
        subplot(2,1,1)
    end
    imagesc(V,[0 1])
    hold all
    plot(B(:,2),B(:,1),'k.')
    colorbar
    text(5,5,['t:' num2str(tfac*t)],'color','k','FontSize',16)     
    axis square
%     for ie=1:n2
%         plot(xel(ie),yel(ie),'ko','MarkerFaceColor',col{ie})
%         text(xel(ie),yel(ie),num2str(ie),'FontSize',16)
%     end
    hold off
    
    % calculate phie
    for k=1:n2
        [matx,maty]=meshgrid((2:X-1)-xel(k),(2:Y-1)-yel(k));
        phie(k,tt)=-sum(sum(du(2:X-1,2:Y-1)'./sqrt(matx.^2+maty.^2))); 
    end
    
    if movieflag==1
        for j=1:n2
            subplot(n2*2,1,n2+j)
            plot(tini:t,squeeze(phie(j,1:tt)),'-','LineWidth',2,'Color',col{j})
            hold all
            ylabel([num2str(j)],'Color',col{j})
            set(gca,'YTick',[])
            xlim([tini t+1])
    %                     ylim([-1 1])
            set(gca,'YTick',-yval*5:yval*5:yval*5)

            if j~=n2
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
        matdisp=reshape(ph,[n n]);
        imagesc(matdisp,[-yval yval])
        hold all
        plot(B(:,2)/10,B(:,1)/10,'k.')
        hold off
%         axis off
        axis square
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