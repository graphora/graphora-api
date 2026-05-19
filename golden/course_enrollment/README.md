# course_enrollment

A short academic-enrollment summary. Tests one instructor
teaching multiple courses + one student taking multiple
courses.

## What this exercises

- **Cross-pattern: bipartite matching**: Two students-side
  edges (ENROLLED_IN) and two instructor-side edges
  (TAUGHT_BY) form a 2x2 bipartite shape. Pin that the
  edge-matching layer handles the four-edge balanced
  structure.
- **Anonymous student identity**: ``STU-77103`` is the
  student's identifier, not name. Pin alongside
  patient_admission's MRN for the broader anonymous-ID
  coverage.
- **Course codes with department prefix**: ``CS-401``,
  ``MATH-220`` — short identifiers with subject prefix.

## Failure signals

- An extractor that creates separate Instructor nodes per
  course taught (treating role-in-course as identity)
  drops Instructor precision.
- An extractor that merges CS-401 and MATH-220 on the
  shared instructor inflates Course collapses.

## Intentional non-extractions

- "Distributed Systems", "Linear Algebra II", "spring 2026
  semester", "this semester" are content the ontology does
  not model — course titles ride as ontology-external
  metadata.
