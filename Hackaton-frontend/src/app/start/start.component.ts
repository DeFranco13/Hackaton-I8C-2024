import { Component } from '@angular/core';
import { ApiService } from '../service/api.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.css']
})
export class StartComponent {
  selectedFile: File | null = null;
  fileResponse = false;

  constructor(private apiService: ApiService, private router: Router) {}

  onFileSelected(event: any): void {
    const file = this.selectedFile = event.target.files[0];
    if (file) {
      const fileLabel = document.getElementById('fileLabel');
      if (fileLabel) {
        fileLabel.textContent = file.name;
      }
      
    }
  }

  redirectUser(){
    this.router.navigate(['/result'])
  }

  changeResponse(antwoord: string){
    if (antwoord == "y"){
      this.fileResponse = true
    }
    else{
      this.fileResponse = false
    }
  }

  onUpload(): void {
    if (this.selectedFile) {
      this.apiService.uploadFile(this.selectedFile).subscribe(
        response => console.log('Upload success!', response),
        error => console.error('Error uploading file!', error)
      );
      this.changeResponse("y")
    } else {
      console.error('No file selected!');
      this.changeResponse("n")
    }
  }
}
