library(ggseqlogo)
library(ggplot2)
library(scales)

extract_splice_site_sequences <- function(classAS_output, original_sequences) {

  splice_sequences <- list(
    "RI donor" = character(), "RI acceptor" = character(),
    "SE donor" = character(), "SE acceptor" = character(),
    "A5 donor" = character(), "A5 acceptor" = character(),
    "A3 donor" = character(), "A3 acceptor" = character()
  )
)

  for (line in classAS_output) {
    fields <- strsplit(line, "\t")[[1]]
    full_seq_name <- fields[1]
    event_type <- fields[2]
    positions <- fields[4]
    start_end <- as.integer(unlist(strsplit(positions, "-")))
    start <- start_end[1]
    end <- start_end[2]

    seq_name <- strsplit(full_seq_name, "\\+")[[1]][1]

    if (seq_name %in% names(original_sequences)) {
      seq_length <- nchar(original_sequences[[seq_name]])

      # upstream 5bp, downstream 4bp
      donor_start <- max(0, start - 5)
      donor_end <- min(seq_length, start + 5)
      acceptor_start <- max(0, end - 5)
      acceptor_end <- min(seq_length, end + 5)

      # 根据事件类型提取相应的序列
      donor_seq <- substr(original_sequences[[seq_name]], donor_start + 1, donor_end)
      acceptor_seq <- substr(original_sequences[[seq_name]], acceptor_start + 1, acceptor_end)

      if (nchar(donor_seq) == 10) {
        splice_sequences[[paste0(event_type, " donor")]] <- c(splice_sequences[[paste0(event_type, " donor")]], donor_seq)
      }
      if (nchar(acceptor_seq) == 10) {
        splice_sequences[[paste0(event_type, " acceptor")]] <- c(splice_sequences[[paste0(event_type, " acceptor")]], acceptor_seq)
      }
    }
  }

  return(splice_sequences)
}

# save fasta
save_sequences_to_fasta <- function(splice_sequences, output_dir) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }

  for (seq_type in names(splice_sequences)) {
    fasta_file <- file.path(output_dir, paste0(seq_type, "_splice_site.fasta"))
    fasta_lines <- character() #
    for (i in seq_along(splice_sequences[[seq_type]])) {
      fasta_lines <- c(fasta_lines, paste0(">", seq_type, "_", i))
      fasta_lines <- c(fasta_lines, splice_sequences[[seq_type]][i])
    }
    writeLines(fasta_lines, fasta_file)
  }
}

save_frequencies_to_csv <- function(seqs, output_dir, seq_type) {
  seq_matrix <- do.call(rbind, strsplit(seqs, ""))

  # evaluate prob
  frequency_matrix <- apply(seq_matrix, 2, function(col) {
    tab <- table(factor(col, levels = c("A", "C", "G", "T")))
    tab / length(col)
  })

  frequency_df <- as.data.frame(t(frequency_matrix))
  colnames(frequency_df) <- c("A", "C", "G", "T")
  frequency_df$Position <- 1:nrow(frequency_df)

  max_base <- apply(frequency_matrix, 2, function(col) {
    base_names <- names(col)
    max_base_index <- which.max(col)
    base_names[max_base_index]
  })

  frequency_df$Most_Common_Base <- max_base

  csv_path <- file.path(output_dir, paste0(seq_type, "_base_frequencies.csv"))
  write.csv(frequency_df, csv_path, row.names = FALSE)

  print(paste("Base frequencies and most common bases saved for", seq_type, "to", csv_path))
}

plot_motif_logos_with_ggseqlogo <- function(splice_sequences, output_dir) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }

  for (seq_type in names(splice_sequences)) {
    if (length(splice_sequences[[seq_type]]) > 1) {  # 确保有足够的序列绘制图
      seqs <- splice_sequences[[seq_type]]

      # 如果seq_type为A3或A5，将其更改为A3SS或A5SS
      title_seq_type <- seq_type
      if (grepl("^A3", seq_type)) {
        title_seq_type <- gsub("^A3", "A3SS", seq_type)
      } else if (grepl("^A5", seq_type)) {
        title_seq_type <- gsub("^A5", "A5SS", seq_type)
      }

      #
      p_prob <- ggseqlogo(seqs, method = "prob") +
        ggtitle(paste(title_seq_type)) +
        theme(
          plot.title = element_text(hjust = 0.5, size = 16),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14)
        )

      prob_png_path <- file.path(output_dir, paste0(seq_type, "_logo_prob.png"))
      prob_tiff_path <- file.path(output_dir, paste0(seq_type, "_logo_prob.tif"))
      prob_svg_path <- file.path(output_dir, paste0(seq_type, "_logo_prob.svg"))
      prob_pdf_path <- file.path(output_dir, paste0(seq_type, "_logo_prob.pdf"))

      ggsave(prob_png_path, plot = p_prob, device = "png")
      ggsave(prob_tiff_path, plot = p_prob, device = "tiff")
      ggsave(prob_svg_path, plot = p_prob, device = "svg")
      pdf(prob_pdf_path, width = 8, height = 6)
      print(p_prob)
      dev.off()


      p_bits <- ggseqlogo(seqs, method = "bits") +
        ggtitle(paste(title_seq_type)) +
        theme(
          plot.title = element_text(hjust = 0.5, size = 16),
          axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16),
          axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14)
        ) +
        scale_y_continuous(limits = c(0, 2))

      bits_png_path <- file.path(output_dir, paste0(seq_type, "_logo_bits.png"))
      bits_tiff_path <- file.path(output_dir, paste0(seq_type, "_logo_bits.tif"))
      bits_svg_path <- file.path(output_dir, paste0(seq_type, "_logo_bits.svg"))
      bits_pdf_path <- file.path(output_dir, paste0(seq_type, "_logo_bits.pdf"))

      ggsave(bits_png_path, plot = p_bits, device = "png")
      ggsave(bits_tiff_path, plot = p_bits, device = "tiff")
      ggsave(bits_svg_path, plot = p_bits, device = "svg")
      pdf(bits_pdf_path, width = 8, height = 6)
      print(p_bits)
      dev.off()


      save_frequencies_to_csv(seqs, output_dir, seq_type)

      print(paste("Logo generated for", seq_type))
    } else {
      print(paste("Warning: Not enough sequences to generate a", seq_type, "motif logo."))
    }
  }
}

main <- function(classAS_output_file, original_sequences_file, output_dir) {


  classAS_output <- readLines(classAS_output_file)

  original_sequences <- list()
  lines <- readLines(original_sequences_file)
  seq_name <- NULL
  for (line in lines) {
    if (startsWith(line, ">")) {
      seq_name <- substring(line, 2)
      original_sequences[[seq_name]] <- ""
    } else {
      original_sequences[[seq_name]] <- paste0(original_sequences[[seq_name]], line)
    }
  }


  splice_sequences <- extract_splice_site_sequences(classAS_output, original_sequences)


  save_sequences_to_fasta(splice_sequences, output_dir)

  plot_motif_logos_with_ggseqlogo(splice_sequences, output_dir)
}


args <- commandArgs(trailingOnly = TRUE)
main(args[1], args[2], args[3])
