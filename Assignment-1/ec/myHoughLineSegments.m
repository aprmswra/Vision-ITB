function [lines] = myHoughLineSegments(lineRho, lineTheta, Im)
    % Define lines structure
    lines = struct('start', {}, 'stop', {});
    
    % For each rho and theta pair
    for i = 1:length(lineRho)
        rho = lineRho(i);
        theta = lineTheta(i);
        
        % Define the start and end points
        start_pnt = [];
        end_pnt = [];
        
        % Vary t to get points on the line
        for t = -max(size(Im)):max(size(Im))
            x = round(rho * cos(theta) + t * sin(theta));
            y = round(rho * sin(theta) - t * cos(theta));
            
            % Check if the point is within the image boundaries
            if x > 0 && x <= size(Im, 2) && y > 0 && y <= size(Im, 1)
                % Check if the point lies on an edge
                if Im(y, x) > 0
                    if isempty(start_pnt)
                        start_pnt = [x, y];
                    end
                    end_pnt = [x, y];
                end
            end
        end
        
        % Store the start and end points in the lines structure
        if ~isempty(start_pnt) && ~isempty(end_pnt)
            lines(end+1).start = start_pnt;
            lines(end).stop = end_pnt;
        end
    end
end
