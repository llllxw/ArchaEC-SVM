# Data sources and attribution

The protein sequences and annotations originate from **The UniProt Consortium**, UniProtKB/Swiss-Prot release **2026_02** (release header: 10 June 2026), downloaded **19 July 2026** using the UniProt REST stream endpoint.

- Source: https://www.uniprot.org/
- API: https://rest.uniprot.org/uniprotkb/stream
- Query: `(reviewed:true) AND (taxonomy_id:2157) AND (ec:*)`
- License: https://www.uniprot.org/help/license
- License deed: https://creativecommons.org/licenses/by/4.0/
- Frozen raw-table SHA-256: `50b94d43d607f047a046211dc880336a177d7d584f5335edbf110fd43fd157f7`

UniProt applies CC BY 4.0 to the copyrightable parts of its databases. Attribution is retained here and in the source manifest. Derived changes include EC normalization, seven binary top-level labels, annotation-count flags, taxonomic descriptors, sequence hashes, numerical sequence features and grouped partition assignments. The release does not imply UniProt endorsement. Links and third-party tools retain their own terms.

The original local source path in the download manifest has been replaced with a portable repository-relative path. The raw TSV, master table, features, selected current results, and archived Python source are copied byte-for-byte and recorded in `docs/source_file_manifest.json`.

The older small FASTA collection and its outdated summary tables are not the released training dataset. Cached dependencies, temporary MMseqs2 databases, model binaries, archived figures, manuscript drafts, and author-local paths are not required for using the released benchmark and are not intentionally published as data assets.
