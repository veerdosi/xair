export interface TreeNode {
  id: string;
  label: string;
  children: TreeNode[];
  type: 'actual' | 'counterfactual';
  impact?: number;
}

export interface Counterfactual {
  id: string;
  summary: string;
  impact: number;
  applied: boolean;
}