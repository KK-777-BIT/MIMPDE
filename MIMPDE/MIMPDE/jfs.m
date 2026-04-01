function model = jfs(type,feat,label,opts)
switch type
  % Ablation
  case 'mimpde-a'  ; fun = @MIDMDE_a;% Ablation: Remove grouped initialization
  case 'mimpde-b'  ; fun = @gMIDMDE_b; % Ablation: Remove dynamic resource allocation
    case 'mimpde-c'  ; fun = @gMIDMDE_c; % Ablation: Remove dominant-slave information sharing
  % Comparison algorithm
  case 'mimpde'  ; fun = @gMIDMDE;
  case 'mi_mpode'  ; fun = @gMI_MPODE;
  case 'mspso'     ; fun = @gMSPSO;  
  case 'tlpso'     ; fun = @gTLPSO;
  case 'ecsa'      ; fun = @gECSA;
  case 'pltvaciwpso'; fun = @gPLTVACIW_PSO;
end
tic;
model = fun(feat,label,opts); 
% Computational time
t = toc;

model.t = t;
% fprintf('\n Processing Time (s): %f % \n',t); fprintf('\n');
end
