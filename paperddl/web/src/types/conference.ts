export interface Conference {
  title: string;
  description: string;
  sub: string;
  dblp?: string;
  rank: {
    ccf: string;
    core?: string;
    thcpl?: string;
    thc?: string;
  };
  acc_str?: string;
  subname?: string;
  subname_en?: string;
  nextDeadline: string | null;
  timezone?: string;
  confs: {
    year: number;
    id: string;
    link: string;
    timeline: {
      abstract_deadline?: string;
      deadline: string;
      comment?: string;
    }[];
    timezone?: string;
    date?: string;
    place?: string;
    note?: string;
  }[];
}

export interface Category {
  name: string;
  sub: string;
}

export interface ConferenceQuery {
  sub?: string[] | string;
  q?: string;
}
