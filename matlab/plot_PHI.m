clc
close all
clear all

% for num = [25000:5000:40000]
for num = 0:1000:20000
    num
    f=figure;
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

    boundary = [0,1*2^10];
    totElement = length(xmesh)/4;
    tick = 0:100:2^11;
    h = tight_subplot(1,1,[0 0]);
    
    
    % PHI
    for i=1:totElement
        a = i*4-3;
        b = i*4;
        plot3(h(1),xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
            'Color',[0.3,0.3,0.3],'LineWidth',1.5);
        hold(h(1),'on')
    end
    scatter3(h(1),x,y,phi,[],phi,'.');
    caxis(h(1),[-1 1])
    axis(h(1),'off')
    xlim(h(1),boundary)
    ylim(h(1),boundary)
    view(h(1),[0,90])

%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot3(h(2),xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
%             'Color',[0.3,0.3,0.3],'LineWidth',1.5);
%         hold(h(2),'on')
%     end
%     scatter3(h(2),x,y,phi,[],phi,'.');
%     caxis(h(2),[-1 1])
%     axis(h(2),'off')
%     xlim(h(2),boundary)
%     ylim(h(2),boundary)
%     view(h(2),[0,90])
%     
%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot3(h(3),xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
%             'Color',[0.3,0.3,0.3],'LineWidth',1.5);
%         hold(h(3),'on')
%     end
%     scatter3(h(3),x,y,phi,[],phi,'.');
%     caxis(h(3),[-1 1])
%     axis(h(3),'off')
%     xlim(h(3),boundary)
%     ylim(h(3),boundary)
%     view(h(3),[180,90])
%     
%     for i=1:totElement
%         a = i*4-3;
%         b = i*4;
%         plot3(h(4),xmesh([a:b,a]),ymesh([a:b,a]),-2*ones(1,5),'-', ...
%             'Color',[0.3,0.3,0.3],'LineWidth',1.5);
%         hold(h(4),'on')
%     end
%     scatter3(h(4),x,y,phi,[],phi,'.');
%     caxis(h(4),[-1 1])
%     axis(h(4),'off')
%     xlim(h(4),boundary)
%     ylim(h(4),boundary)
%     view(h(4),[90,90])
    
    set(gcf,'PaperUnits','inches');
    set(gcf,'PaperSize',[8 8]);
    set(gcf,'PaperPosition',[0.5 0.5 7 7]);
    set(gcf,'PaperPositionMode', 'Manual');
    saveas(gca,['PHI_',num2str(num)],'jpg');
%     print(['PHI_',num2str(num)],'-djpg','-r1000','-painters');

    close all
end
% close all
