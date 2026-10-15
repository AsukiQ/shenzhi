export interface PaperRecord {
  id: number;
  paper_id: string;
  arxiv_id: string;
  source: string;
  conference: string;
  year: number;
  subjects: string[];
  primary_subject: string;
  track: string;
  title: string;
  authors: string[];
  abstract: string;
  pdf_url: string;
  keywords: string[];
  type: string;
  editor: string[];
  booktitle: string;
  pages: string;
  publisher: string;
  doi: string;
  biburl: string;
  bibsource: string;
  source_file: string;
}

export interface PaperSearchResult {
  items: PaperRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface PaperQuery {
  q?: string;
  conference?: string;
  year?: string | number;
  track?: string;
  limit?: number;
  offset?: number;
}

export interface PaperTrack {
  year: number;
  track: string;
  count: number;
}

export interface PaperVenue {
  conference: string;
  count: number;
  minYear: number;
  maxYear: number;
  yearCount: number;
  trackCount: number;
  years: number[];
}
