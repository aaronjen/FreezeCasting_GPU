clc
close all
clear all

% for num = [25000:5000:40000]
for num = 15000
    num
    x = load(['outX_',num2str(num)]);
    y = load(['outY_',num2str(num)]);
    phi = load(['outPHI_',num2str(num)]);
    u = load(['outU_',num2str(num)]);
    xmesh = load(['meshX_',num2str(num)]);
    ymesh = load(['meshY_',num2str(num)]);

    X   = [0.5*x(:);    0.75*x(:);      x(:)];
    Y   = [0.5*y(:);    0.75*y(:);      y(:)];
    PHI = [0.5*phi(:);  0.75*phi(:);    phi(:)];
    U   = [0.5*u(:);    0.75*u(:);      u(:)];

    boundary = [0,0.8*2^7];
    totElement = length(xmesh)/4;
    tick = 0:100:2^11;
    
    % PHI
    figure
%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot3(xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
%             'Color',[0.3,0.3,0.3]);
%         hold on
%     end
    hold on
    scatter3(x,y,phi,[],phi,'.');
    colormap;
    colorbar;
    title('PHI')
    view([0,90])
    caxis([-1 1])
    axis 'square'
    xlim(boundary)
    ylim(boundary)
    view([0,90])
    grid off
    set(gca,'TickDir','out','Xtick',tick,'Ytick',tick)
%     print(['PHI_',num2str(num)],'-dpng','-r500');

%     % U
%     figure
%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot3(xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
%             'Color',[0.3,0.3,0.3]);
%         hold on
%     end
%     hold on
%     scatter3(x,y,u,[],u,'.');
%     colormap;
%     colorbar;
%     axis 'square';
%     title('U')
%     view([0,90])
%     caxis([-0.7 0])
%     grid off
%     axis 'square'
%     xlim(boundary)
%     ylim(boundary)
%     view([0,90])
%     set(gca,'TickDir','out','Xtick',tick,'Ytick',tick)
% %     print(['U_',num2str(num)],'-dpng','-r500');
% 
%     % mesh
%     figure
%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot(xmesh([a:b,a]),ymesh([a:b,a]),'-k');
%         hold on
%     end
%     title('Mesh')
%     axis 'square'
%     xlim(boundary)
%     ylim(boundary)
%     set(gca,'TickDir','out','Xtick',tick,'Ytick',tick)
% %     print(['Mesh_',num2str(num)],'-dpng','-r500');
% 
% %     close all
end
% close all
